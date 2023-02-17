from options import parse_args
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import openslide
import os
import math
import dgl
from sklearn.neighbors import kneighbors_graph
from dgl.data.utils import save_graphs
from histocartography.visualization import OverlayGraphVisualization
import warnings

warnings.filterwarnings('ignore')
np.random.seed(123)

CellTypes = ['Tumor cells',
             'Vascular endothelial cell',
             'Lymphocytes',
             'Fibroblast',
             'Biliary epithelial cell',
             'Hepatocyte',
             'Other']

CellTypeLabels = {'Tumor cells': 1,
                  'Vascular endothelial cell': 2,
                  'Lymphocyte': 3,
                  'Fibroblast': 4,
                  'Biliary epithelial cell': 5,
                  'Hepatocyte': 6,
                  'Plasmacyte': 7,
                  'Other': 8}

SelectedFeatures = ['Class', 'Centroid X µm','Centroid Y µm',
                    'Nucleus: Hematoxylin OD mean', 'Nucleus: Hematoxylin OD std dev', 'Nucleus: Eosin OD mean', 'Nucleus: Eosin OD std dev',
                    'Cell: Area', 'Cell: Circularity', 'Nucleus/Cell area ratio', 'Cell: Hematoxylin OD std dev', 'Cell: Eosin OD std dev',
                    'Cytoplasm: Hematoxylin OD mean', 'Cytoplasm: Eosin OD mean']


def load_txt(LabelDataPath, Patient):
    """
    加载当前patient对应的qupath计算得到的相关细胞参数归一化结果
    :param LabelDataPath: 保存所有标记结果工程的总文件夹
    :param Patient: 当前患者编号
    :return: DataFrame类型表格
    """
    inTXTDataPath = os.path.join(LabelDataPath, Patient, Patient + '.txt')
    inTXTData = pd.read_csv(inTXTDataPath, sep='\t', engine='python')
    inTXTData['Class'] = inTXTData['Class'].replace('Tumor', 'Tumor cells')
    inTXTData['Class'] = inTXTData['Class'].replace('Lymphocytes', 'Lymphocyte')
    inTXTData['Class'] = inTXTData['Class'].replace('Vasular endothelial cell', 'Vascular endothelial cell')
    inTXTData['Class'] = inTXTData['Class'].replace('others', 'Other')
    inTXTData['Class'] = inTXTData['Class'].replace('Others', 'Other')
    inTXTData['Class'] = inTXTData['Class'].replace('other', 'Other')
    inTXTData['Class'] = inTXTData['Class'].replace('Hepatpcytes', 'Hepatocyte')
    inTXTData['Class'] = inTXTData['Class'].replace('Hepayocyte', 'Hepatocyte')
    inTXTData['Class'] = inTXTData['Class'].replace('Billiary epithelial cell', 'Biliary epithelial cell')
    print('Finished load_txt')
    return inTXTData


def get_range_for_every_feature(LabelDataPath):
    """
    确定所有患者txt表格中每个指标的上下限, 用于后面归一化操作
    :param LabelDataPath: 保存标记工程的总文件
    :return: (2 * 41)数据, 第一维为min 第二维为max, 每一列为一个指标
    """
    Patients = os.listdir(LabelDataPath)
    MinAndMax = np.empty(shape=(2, 41))
    MinAndMax[0, :] = 10000
    MinAndMax[1, :] = -10000
    for Patient in Patients:
        TXTData = load_txt(LabelDataPath, Patient)
        Features = TXTData.iloc[:, 7:]
        for i in range(41):
            if Features.min()[i] < MinAndMax[0, i]:
                MinAndMax[0, i] = Features.min()[i]
            if Features.max()[i] > MinAndMax[1, i]:
                MinAndMax[1, i] = Features.max()[i]

    print('Finished get_range_for_every_feature')
    return MinAndMax


def load_image(LabelDataPath, Patient):
    """
    加载当前patient对应的细胞标注结果图和细胞核结果图
    :param LabelDataPath: 保存所有标记结果工程的总文件夹
    :param Patient: 当前患者编号
    :return: Numpy数组格式细胞标注图和细胞核标注图
    """
    Image.MAX_IMAGE_PIXELS = None
    LabeledCellImagePath = os.path.join(LabelDataPath, Patient, Patient + '-CellLabels.png')
    LabeledCellImage = Image.open(LabeledCellImagePath)
    LabeledCellImage = np.array(LabeledCellImage)
    LabeledNucleiImagePath = os.path.join(LabelDataPath, Patient, Patient + '-NucleiLabels.png')
    LabeledNucleiImage = Image.open(LabeledNucleiImagePath)
    LabeledNucleiImage = np.array(LabeledNucleiImage)

    print('Finished load_image')
    return LabeledCellImage, LabeledNucleiImage


def load_wsi(WSIDataPath, Patient):
    """
    加载当前patient对应WSI数据
    :param WSIDataPath: 保存所有患者WSI数据的总文件夹
    :param Patient: 当前患者编号
    :return: openslide格式的slide数据，可使用np.array(slide.read_region函数读取图片转化为numpy数组)
    """
    path = os.path.join(WSIDataPath, Patient, Patient + '.ndpi')
    slide = openslide.OpenSlide(path)

    print('Finished load_wsi')
    return slide


def find_patch_boxes(LabeledCellImage, LabeledTissueImage, patch_size=512, ratio=2000000):
    """

    :param LabeledCellImage: 细胞标注图像，numpy数组格式
    :param LabeledTissueImage: 组织标注图像，numpy数组格式
    :param patch_size: 截取一个patch的边长，默认为512像素
    :param ratio: tissue面积与patch数目的比值，默认为2,000,000
    :return: list格式的截取的patch的BoundingBox坐标(xmin, ymin, xmax, ymax)
    """
    # 初始化参数
    SelectedBoxList = []
    # 选择包含细胞种类最多的Patches
    for i in np.unique(LabeledTissueImage):
        if i > 0:
            # 提取同一种类的Tissue，计算面积和当前Tissue的patch数目n
            SubImage = np.array(LabeledTissueImage == i, np.uint8)
            S = np.sum(SubImage)
            n = math.ceil(S / ratio)
            # 确定滑窗扫描的区域
            contours, _ = cv2.findContours(SubImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundingBoxes = np.array([cv2.boundingRect(c) for c in contours])
            xmin = np.min(boundingBoxes[:, 0])
            xmax = np.max(boundingBoxes[:, 0] + boundingBoxes[:, 2])
            ymin = np.min(boundingBoxes[:, 1])
            ymax = np.max(boundingBoxes[:, 1] + boundingBoxes[:, 3])
            boundingBox = [xmin, ymin, xmax, ymax]
            # 列举滑窗BoundingBox坐标
            box_list = []
            for i in np.arange(xmin, xmax - patch_size, patch_size):
                for j in np.arange(ymin, ymax - patch_size, patch_size):
                    box = (i, j, i + patch_size, j + patch_size)
                    box_list.append(box)
            # 记录每个滑窗中CellType的数目
            CellTypeNum = []
            for box in box_list:
                patch_xmin, patch_ymin, patch_xmax, patch_ymax = box
                patch = LabeledCellImage[patch_xmin: patch_xmax, patch_ymin: patch_ymax]
                CellTypeNum.append(len(np.unique(patch)) - 1)
            # 选取的Patch中细胞数目是最多的
            CellTypeNum = np.array(CellTypeNum)
            index = np.where(CellTypeNum == np.max(CellTypeNum))[0]
            # 产生n个随机数，及随机取n个patch
            selected_index = []
            for i in np.random.randint(0, len(index), n):
                selected_index.append(index[i])

            for index in selected_index:
                box = box_list[index]
                SelectedBoxList.append(box)
    SelectedBoxList = list(set(SelectedBoxList))

    print('Finished find_patch_boxes')
    return SelectedBoxList


def find_patch_boxes_new(LabeledCellImage, patch_size=512, ratio=4000000):
    """
    由细胞标记图像提取细胞种类最多的Patch，
    :param LabeledCellImage: 细胞标记图像
    :param patch_size: patch边长
    :param ratio: patch个数/总面积
    :return:
    """
    # 分割出前景
    CellImage = ((LabeledCellImage > 0) * 255).astype('uint8')
    close_kernel = np.ones((50, 50), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(CellImage), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((50, 50), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    # 计算总面积
    image_binary = np.array(image_open > 0, dtype='int')
    S = image_binary.sum()

    # 获得前景的Bounding Box
    contours, _ = cv2.findContours(image_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = np.array([cv2.boundingRect(c) for c in contours])

    xmin = np.min(boundingBoxes[:, 0])
    xmax = np.max(boundingBoxes[:, 0] + boundingBoxes[:, 2])
    ymin = np.min(boundingBoxes[:, 1])
    ymax = np.max(boundingBoxes[:, 1] + boundingBoxes[:, 3])

    # 滑窗获得每个patch的BoundingBox坐标
    box_list = []
    for i in np.arange(xmin, xmax - patch_size, patch_size):
        for j in np.arange(ymin, ymax - patch_size, patch_size):
            # 注意到cv2 x和y坐标颠倒问题
            box = (i, j, i + patch_size, j + patch_size)
            box_list.append(box)

    # 统计每个patch内细胞种类数
    CellTypeNum = []
    for box in box_list:
        # 这里的x和y就是图像对应的竖直方向和水平方向
        patch_xmin, patch_ymin, patch_xmax, patch_ymax = box
        patch = LabeledCellImage[patch_ymin: patch_ymax, patch_xmin: patch_xmax]
        CellTypeNum.append(len(np.unique(patch)) - 1)

    box_list = pd.DataFrame(box_list)
    CellTypeNum = pd.DataFrame(CellTypeNum)

    # 依据每个patch中不同种类细胞数目排序，取前
    boxes = pd.concat([CellTypeNum, box_list], axis=1)
    boxes.columns = ['CellTypeNum', 'x_min', 'y_min', 'x_max', 'y_max']
    boxes = boxes.sort_values(by='CellTypeNum', ascending=False)

    # 一张切片最少10个patch，最多50个patch
    num = int(S / ratio)
    num = np.clip(num, 10, 50)

    # 如果num超过patch总数目，则从所有patch内随机取（避免超出引用范围报错）
    if num < boxes.shape[0]:
        boxes_to_be_selected = boxes[boxes['CellTypeNum'] >= boxes.iloc[num, 0]]
    else:
        boxes_to_be_selected = boxes

    # 从待选patch中随机抽取num个patch
    boxes_selected = boxes_to_be_selected.sample(n=num, random_state=123, axis=0)
    SelectedBoxList = np.array(boxes_selected.iloc[:, 1:])
    print('Finished find_patch_boxes')
    return SelectedBoxList


def get_origin_image(WSIData, level_dimensions):
    """
    从slide数据读出降采样level为level_dimensions的RGB图片，输出为numpy数组格式
    :param WSIData: slide数据
    :param level_dimensions: 降采样层级，取值[0-8]
    :return: numpy数组RGB图片
    """
    (m, n) = WSIData.level_dimensions[level_dimensions]
    OriginImage = np.array(WSIData.read_region((0, 0), level_dimensions, (m, n)))[:, :, :3]

    print('Finished get_origin_image')
    return OriginImage


def get_cell_coordinate_pixel(CellData, box):
    """
    将CellData表格中坐标信息转化为当前patch中对应的像素坐标
    :param CellData: 当前patch中的细胞表格
    :param box: 当前patch对应的box, 单位为微米
    :return: 列表格式各细胞对应像素坐标
    """
    CoordinateUm = [list(CellData['Centroid X µm']), list(CellData['Centroid Y µm'])]
    CoordinatePixelX = ((CoordinateUm[0] - box[0]) / (Ratio * 4)).astype('int')
    CoordinatePixelY = ((CoordinateUm[1] - box[1]) / (Ratio * 4)).astype('int')
    CoordinatePixel = []
    for i in range(len(CoordinatePixelX)):
        CoordinatePixel.append((CoordinatePixelX[i], CoordinatePixelY[i]))
    CoordinatePixel = list(set(CoordinatePixel))
    return CoordinatePixel


def generate_graph(Centroids, Features, PatchSize=512, k=5, thresh=30):
    """
    以KNN方式构建图结构
    :param Centroids: 图节点的坐标, list
    :param Features: 所有节点的特征向量, tensor
    :param PatchSize: 当前Patch大小, int
    :param k: KNN中K
    :param thresh: 超过threshold不再建立边
    :return: dgl.graph
    """
    graph = dgl.DGLGraph()
    graph.add_nodes(len(Centroids))

    image_size = (PatchSize, PatchSize)
    # 设置图节点中心坐标
    graph.ndata['centroid'] = torch.FloatTensor(Centroids)
    # 设置图节点特征（特征还包括归一化的坐标信息）
    Features[:, 1] = Features[:, 1] / image_size[0]
    Features[:, 2] = Features[:, 2] / image_size[1]
    graph.ndata['feat'] = Features

    # 利用KNN方法构建图
    centroids = graph.ndata['centroid']
    if Features.shape[0] != 1:
        k = min(Features.shape[0] - 1, k)
        adj = kneighbors_graph(
            centroids,
            k,
            mode="distance",
            include_self=False,
            metric="euclidean").toarray()

        if thresh is not None:
            adj[adj > thresh] = 0

        edge_list = np.nonzero(adj)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))

    return graph


def generate_and_save_cell_graphs(box, TXTData, Label, OutPath):
    """
    对应于每个box, 将Patch内所有细胞构建cell-graph, 并保存在Graph文件中
    :param box: patch坐标单位为um
    :param TXTData: 医生标记结果导出的TXT表格
    :param Patient: 当前病患, 在Graph文件夹中建立相应文件夹
    """
    # 将box内的所有细胞筛选出来
    CellINPatch = TXTData[(box[0] < TXTData['Centroid X µm']) & (TXTData['Centroid X µm'] < box[2]) &
                          (box[1] < TXTData['Centroid Y µm']) & (TXTData['Centroid Y µm'] < box[3])]
    CoordinatePixel = get_cell_coordinate_pixel(CellINPatch, box)
    Features = CellINPatch[SelectedFeatures]
    Features['Class'] = Features['Class'].map(CellTypeLabels)
    Features['Centroid X µm'] = ((Features['Centroid X µm'] - box[0]) / Ratio / 4).astype('int')
    Features['Centroid Y µm'] = ((Features['Centroid Y µm'] - box[1]) / Ratio / 4).astype('int')
    Features = Features.drop_duplicates(subset=['Centroid X µm', 'Centroid Y µm'], keep='first')
    Features = np.array(Features, dtype='float64')
    Features = torch.from_numpy(Features)
    Graph = generate_graph(CoordinatePixel, Features)
    GraphPath = OutPath
    save_graphs(GraphPath, Graph, Label)
    return


def graph_visualize(box, WSIImage, graph):
    """
    图结构的可视化，用于检查和超参数的选择
    :param box: patch边界box，微米
    :param WSIImage: 读入的numpy数组格式的原始图像，降采样率：4
    :param graph: 要可视化的graph，dgl.graph
    """
    boxpixel = (box / Ratio / 4).astype('int')
    image = WSIImage[boxpixel[1]: boxpixel[3], boxpixel[0]: boxpixel[2]]
    visualizer = OverlayGraphVisualization(node_radius=1,
                                           edge_thickness=1,
                                           )
    canvas = visualizer.process(image, graph)
    canvas.show()


def show_big_array(array):
    """
    展示非常大数组图片（避免plt导致python无响应）
    :param array: 大数组
    """
    p = Image.fromarray(array)
    p.show()


if __name__ == '__main__':
    opt = parse_args()
    # 载入数据
    LabelDataPath = opt.label_data_path  # 保存所有标记结果工程的总文件夹
    WSIDataPath = opt.WSI_data_path  # 保存所有WSI数据的总文件夹
    FollowUpData = pd.read_csv(opt.follow_up_data, sep='\t', engine='python')  # 加载随访数据
    FollowUpData['标本号'] = FollowUpData['标本号'].astype(str)
    FollowUpData.dropna(axis=0, how='all')

    # 带区域标记的病人
    Patients = os.listdir(LabelDataPath)
    Patients.sort()
    # FeaturesMinAndMax = np.loadtxt('MinAndMax.csv')

    for Patient in tqdm(Patients):
        if os.path.exists(os.path.join(opt.graph_out_path, Patient)) is False:
            os.makedirs(os.path.join(opt.graph_out_path, Patient))
            TXTData = load_txt(LabelDataPath, Patient)  # 加载TXT表格
            print(set(TXTData['Class']) - CellTypeLabels.keys())
            LabeledCellImage, LabeledNucleiImage = load_image(LabelDataPath, Patient)  # 加载标注图片
            WSIData = load_wsi(WSIDataPath, Patient)  # 加载WSI数据
            # WSIImage = get_origin_image(WSIData, level_dimensions=2)

            # 选择包含细胞种类最多的Patches
            SelectedPatchBoxes = find_patch_boxes_new(LabeledCellImage)
            SelectedPatchBoxes = np.array(SelectedPatchBoxes)

            # 构建图结构
            Ratio = float(WSIData.properties['openslide.mpp-x'])

            SelectedPatchBoxesUm = SelectedPatchBoxes * Ratio * 4  # 将像素框转换为实际距离

            for box in SelectedPatchBoxesUm:
                BoxName = str(box[0]) + '_' + str(box[1]) + '.bin'
                OutPath = os.path.join(opt.graph_out_path, Patient, 'CellGraph', BoxName)
                PatientFollowUp = FollowUpData[FollowUpData['标本号'] == Patient]
                if not PatientFollowUp.empty:
                    # 复发标签
                    SurvLabel = {'CoxLabel': torch.tensor([(float(PatientFollowUp['无瘤/月']),
                                                            float(PatientFollowUp['复发']))])}
                    # 存活标签
                    # SurvLabel = {'CoxLabel': torch.tensor([(float(PatientFollowUp['生存/月']),
                    #                                         float(PatientFollowUp['死亡']))])}
                    generate_and_save_cell_graphs(box, TXTData, SurvLabel, OutPath)
            print('Finished generate graph for ' + Patient)

        else:
            continue

    # 可视化
    # Graph = dgl.load_graphs(os.path.join(OutPath, 'Tumor cells.bin'))[0][0]
    # graph_visualize(box, WSIImage, Graph)


