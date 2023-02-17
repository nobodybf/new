"""Split data into 3 sets: train, validation and test"""

import os
from sklearn.model_selection import train_test_split
import shutil
from options import parse_args
from glob import glob


def copy_file(old_file_path, new_path):
    """
    复制old_file_path文件到new_path文件夹下
    :param old_file_path: 待移动的文件路径
    :param new_path: 目标文件夹
    """
    file_name = old_file_path.split('\\')[-1]
    src = old_file_path
    dst = os.path.join(new_path, file_name)
    try:
        shutil.copytree(src, dst)
    except:
        shutil.copyfile(src, dst)


def move_file(old_file_path, new_path):
    """
    复制old_file_path文件到new_path文件夹下
    :param old_file_path: 待移动的文件路径
    :param new_path: 目标文件夹
    """
    file_name = old_file_path.split('\\')[-1]
    src = old_file_path
    dst = os.path.join(new_path, file_name)
    shutil.move(src, dst)


def split_on_patient(cg_path, out_path):
    # 获取所有graph地址
    patient_fnames = []
    patients = os.listdir(cg_path)
    for patient in patients:
        patient_path = os.path.join(cg_path, patient)
        patient_fnames.append(patient_path)

    data = patient_fnames
    x_train, x_test = train_test_split(data, test_size=0.2)
    train_target_path = os.path.join(out_path, 'train')
    if not os.path.exists(train_target_path):
        os.makedirs(train_target_path)
    for file in x_train:
        copy_file(file, train_target_path)

    data = x_test
    x_validation, x_test = train_test_split(data, test_size=0.5)
    validation_target_path = os.path.join(out_path, 'validation')
    if not os.path.exists(validation_target_path):
        os.makedirs(validation_target_path)
    for file in x_validation:
        copy_file(file, validation_target_path)

    test_target_path = os.path.join(out_path, 'test')
    if not os.path.exists(test_target_path):
        os.makedirs(test_target_path)
    for file in x_test:
        copy_file(file, test_target_path)

    return


def split_on_patient_version2(cg_path, out_path):
    """
    划分数据集：
    以病人为单位，单独划分出最终用于测试的测试集
    训练集中再进行训练集、验证集和测试集的划分，以patch为单位
    :param cg_path:
    :param out_path:
    :return:
    """
    # 获取所有graph地址
    patient_fnames = []
    patients = os.listdir(cg_path)
    for patient in patients:
        patient_path = os.path.join(cg_path, patient)
        patient_fnames.append(patient_path)

    data = patient_fnames
    x_train, x_test = train_test_split(data, test_size=0.2)

    test_target_path = os.path.join(out_path, 'OverallTest')
    if not os.path.exists(test_target_path):
        os.makedirs(test_target_path)
    for file in x_test:
        copy_file(file, test_target_path)

    data = x_train
    x_train, x_validation = train_test_split(data, test_size=0.25)

    train_target_path = os.path.join(out_path, 'OverallTrain')
    if not os.path.exists(train_target_path):
        os.makedirs(train_target_path)
    for file in x_train:
        copy_file(file, train_target_path)

    validation_target_path = os.path.join(out_path, 'OverallValidation')
    if not os.path.exists(validation_target_path):
        os.makedirs(validation_target_path)
    for file in x_validation:
        copy_file(file, validation_target_path)

    patch_fnames = []
    for patient in x_train:
        patient = patient.split('\\')[-1]
        patch_fnames = patch_fnames + glob(os.path.join(cg_path, patient, '*.bin'))

    data = patch_fnames
    x_train, x_validation = train_test_split(data, test_size=0.4)
    train_target_path = os.path.join(out_path, 'train')
    if not os.path.exists(train_target_path):
        os.makedirs(train_target_path)
    for file in x_train:
        copy_file(file, train_target_path)

    data = x_validation
    x_validation, x_test = train_test_split(data, test_size=0.5)
    validation_target_path = os.path.join(out_path, 'validation')
    if not os.path.exists(validation_target_path):
        os.makedirs(validation_target_path)
    for file in x_validation:
        copy_file(file, validation_target_path)

    validation_target_path = os.path.join(out_path, 'test')
    if not os.path.exists(validation_target_path):
        os.makedirs(validation_target_path)
    for file in x_test:
        copy_file(file, validation_target_path)


def split_on_patch(cg_path, out_path):
    patch_fnames = []
    patients = os.listdir(cg_path)
    for patient in patients:
        patches = os.listdir(os.path.join(cg_path, patient))
        for patch in patches:
            patch_fname = os.path.join(cg_path, patient, patch)
            patch_fnames.append(patch_fname)

    data = patch_fnames
    x_train, x_test = train_test_split(data, test_size=0.2)
    train_target_path = os.path.join(out_path, 'train')
    if not os.path.exists(train_target_path):
        os.makedirs(train_target_path)
    for file in x_train:
        copy_file(file, train_target_path)

    data = x_test
    x_validation, x_test = train_test_split(data, test_size=0.5)
    validation_target_path = os.path.join(out_path, 'validation')
    if not os.path.exists(validation_target_path):
        os.makedirs(validation_target_path)
    for file in x_validation:
        copy_file(file, validation_target_path)

    test_target_path = os.path.join(out_path, 'test')
    if not os.path.exists(test_target_path):
        os.makedirs(test_target_path)
    for file in x_test:
        copy_file(file, test_target_path)

    return


if __name__ == "__main__":
    opt = parse_args()
    cg_path = opt.graph_out_path
    out_path = opt.split_out_path
    split_on_patient_version2(cg_path, out_path)
    # split_on_patch(cg_path, out_path)


    # # 将所有graph按照细胞类型分类
    # graph_by_type = {}
    # for graph_fname in graph_fnames:
    #     cell_type = os.path.split(graph_fname)[-1][: -4]
    #     # 以字典格式储存（key: cell type, value: graph path）
    #     graph_by_type[cell_type] = graph_by_type.get(cell_type, []) + [graph_fname]
    # print('classification done')
    #
    # # 遍历所有细胞种类，对每个细胞种类内的所有graph划分训练集、验证集和测试集(0.8: 0.1: 0.1)
    # # 划分出训练集
    # for key in graph_by_type.keys():
    #     data = graph_by_type[key]
    #     if len(data) != 1:
    #         x_train, x_test = train_test_split(data, test_size=0.2)
    #
    #         train_target_path = os.path.join(out_path, key, 'train')
    #         if not os.path.exists(train_target_path):
    #             os.makedirs(train_target_path)
    #         for file in x_train:
    #             copy_file(file, train_target_path)
    #
    #         test_target_path = os.path.join(out_path, key, 'test')
    #         if not os.path.exists(test_target_path):
    #             os.makedirs(test_target_path)
    #         for file in x_test:
    #             copy_file(file, test_target_path)
    # # 划分出验证集和测试集
    # cell_types = os.listdir(out_path)
    # for cell_type in cell_types:
    #     test_graph_path = os.path.join(out_path, cell_type, 'test')
    #     test_data_names = []
    #     for (dirpath, dirnames, filenames) in os.walk(test_graph_path):
    #         for filename in filenames:
    #             test_data_names += [os.path.join(dirpath, filename)]
    #         new_test, new_validation = train_test_split(test_data_names, test_size=0.5)
    #
    #         new_validation_path = os.path.join(out_path, cell_type, 'validation')
    #         if not os.path.exists(new_validation_path):
    #             os.makedirs(new_validation_path)
    #         for file in new_validation:
    #             move_file(file, new_validation_path)
    # print('split done')



