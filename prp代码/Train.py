import torch
import os
import uuid
import yaml
from tqdm import tqdm
import numpy as np
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from options import parse_args

from histocartography.ml import CellGraphModel, HACTModel
import hiddenlayer as hl
from dataloader import make_data_loader
import warnings
warnings.filterwarnings('ignore')

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

# cuda support
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
NODE_DIM = 14


def CoxLoss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = torch.Tensor(hazard_pred.reshape(-1))
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
    return loss_cox


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazardsdata = hazardsdata.detach().numpy()
    hazards_dichotomize[np.where(hazardsdata > median)[0]] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return pvalue_pred


def CIndex(hazards, labels):
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    for i in range(N_test):
        if labels[i, 1] == 1:
            for j in range(N_test):
                if labels[j, 0] > labels[i, 0]:
                    total += 1
                    if hazards[j] < hazards[i]: concord += 1
                    elif hazards[j] == hazards[i]: concord += 0.5

    return (concord / total)


if __name__ == "__main__":
    opt = parse_args()
    DataPath = opt.split_out_path
    batch_size = 10
    learning_rate = 0.0001
    epochs_num = 50

    # Load config file
    f = open('CellGraphConfig.yml', 'r')
    config = yaml.load(f, Loader=yaml.FullLoader)

    # Set path to save model
    model_path = os.path.join('Model', str(uuid.uuid4()))
    os.makedirs(model_path, exist_ok=True)

    # Make data loaders
    train_dataloader = make_data_loader(
        cg_path=DataPath + '\\train',
        batch_size=batch_size
    )

    val_dataloader = make_data_loader(
        cg_path=DataPath + '\\validation',
        batch_size=batch_size
    )

    test_dataloader = make_data_loader(
        cg_path=DataPath + '\\test',
        batch_size=batch_size
    )

    # Declare model
    model = CellGraphModel(
        gnn_params=config['gnn_params'],
        classification_params=config['classification_params'],
        node_dim=NODE_DIM,
        num_classes=1
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=5e-4
    )

    # Training loop
    step = 0
    best_val_loss = 50
    best_val_cindex = 0
    total_loss_list = []

    for epoch in range(epochs_num):
        loss_list = []
        cindex_list = []
        # A.) Train for 1 epoch
        model = model.to(DEVICE)
        model.train()
        step = -1
        for batch in tqdm(train_dataloader, desc='Epoch training {}'.format(epoch), unit='batch'):
            # 1. Forward pass
            labels = batch[-1]
            data = batch[: -1]
            pred = model(*data)
            cpu_pred = pred.cpu().detach().numpy()

            # 2. Backward pass
            label_reshape = torch.empty((len(labels), 2))
            for i in range(len(labels)):
                label_reshape[i] = labels[i][0]
            label_reshape = label_reshape.to('cuda:0')
            cpu_label = label_reshape.cpu().detach().numpy()
            loss_cox = CoxLoss(label_reshape[:, 0], label_reshape[:, 1], pred, device='cuda:0')
            loss_list.append(loss_cox.cpu().detach().numpy())

            try:
                cindex_train = concordance_index(cpu_label[:, 0], -cpu_pred, cpu_label[:, 1])
                # cindex_train = CIndex(-cpu_pred, cpu_label)
            except:
                cindex_train = np.mean(cindex_list)
            cindex_list.append(cindex_train)

            optimizer.zero_grad()
            loss_cox.backward()
            optimizer.step()
            step += 1
        total_loss_list.append(np.mean(loss_list))
        print('\n')
        print('Train loss {}'.format(np.mean(loss_list)))
        print('Train c-index {}'.format(np.mean(cindex_list)))

        # B.) validate
        model.eval()
        all_val_logits = []
        all_val_labels = []
        for batch in tqdm(val_dataloader, desc='Epoch validation {}'.format(epoch), unit='batch'):
            labels = batch[-1]
            data = batch[:-1]
            label_reshape = torch.empty((len(labels), 2))
            for i in range(len(labels)):
                label_reshape[i] = labels[i][0]
            label_reshape = label_reshape.to('cuda:0')

            with torch.no_grad():
                pred = model(*data)
            all_val_logits.append(pred)
            all_val_labels.append(label_reshape)

        all_val_logits = torch.cat(all_val_logits)
        all_val_preds = all_val_logits
        all_val_labels = torch.cat(all_val_labels)

        # compute & store loss + model
        with torch.no_grad():
            loss = CoxLoss(all_val_labels[:, 0], all_val_labels[:, 1], all_val_preds, device='cuda:0').item()
        if loss < best_val_loss:
            best_val_loss = loss
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_loss.pt'))
        print('Val loss {}'.format(loss))

        # compute c-index
        cpu_label = all_val_labels.cpu()
        cpu_pred = all_val_preds.cpu()
        cindex_validate = concordance_index(cpu_label[:, 0], -cpu_pred, cpu_label[:, 1])
        if cindex_validate > best_val_cindex:
            best_val_cindex = cindex_validate
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_cindex.pt'))
        # cindex_validate = CIndex(all_val_preds, all_val_labels)
        print('Val c-index {}'.format(cindex_validate))

        if epoch % 100 == 0:
            checkpoint_path = 'checkpoints\\step_%d.pt' % epoch
            checkpoint = {
                'model': model.state_dict(),
                'opt': opt
            }
            torch.save(checkpoint, checkpoint_path)

    # draw train loss curve
    plt.figure()
    plt.plot(total_loss_list)
    plt.savefig('train_loss.png')

    # testing loop
    model.eval()
    metric = 'best_val_loss'
    print('\n*** Start testing {} model ***'.format(metric))

    model_name = [f for f in os.listdir(model_path) if f.endswith(".pt") and metric in f][0]
    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))

    all_test_logits = []
    all_test_labels = []
    for batch in tqdm(test_dataloader, desc='Testing: {}'.format(metric), unit='batch'):
        labels = batch[-1]
        data = batch[:-1]

        label_reshape = torch.empty((len(labels), 2))
        for i in range(len(labels)):
            label_reshape[i] = labels[i][0]
        label_reshape = label_reshape.to('cuda:0')

        with torch.no_grad():
            logits = model(*data)
        all_test_logits.append(logits)
        all_test_labels.append(label_reshape)

    all_test_logits = torch.cat(all_test_logits).cpu()
    all_test_preds = all_test_logits
    all_test_labels = torch.cat(all_test_labels).cpu()

    # compute & store loss
    with torch.no_grad():
        loss = CoxLoss(all_test_labels[:, 0], all_test_labels[:, 1], all_test_preds, device='cpu').item()
        cindex_test = concordance_index(all_test_labels[:, 0], -all_test_preds, all_test_labels[:, 1])
        pvalue_test = cox_log_rank(all_test_preds, all_test_labels[:, 1], all_test_labels[:, 0])
    print('Test loss {}'.format(loss))
    print('Test c-index {}'.format(cindex_test))
    print('Test p-value {}'.format(pvalue_test))

    model.eval()
    metric = 'best_val_cindex'
    print('\n*** Start testing {} model ***'.format(metric))

    model_name = [f for f in os.listdir(model_path) if f.endswith(".pt") and metric in f][0]
    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))

    all_test_logits = []
    all_test_labels = []
    for batch in tqdm(test_dataloader, desc='Testing: {}'.format(metric), unit='batch'):
        labels = batch[-1]
        data = batch[:-1]

        label_reshape = torch.empty((len(labels), 2))
        for i in range(len(labels)):
            label_reshape[i] = labels[i][0]
        label_reshape = label_reshape.to('cuda:0')

        with torch.no_grad():
            logits = model(*data)
        all_test_logits.append(logits)
        all_test_labels.append(label_reshape)

    all_test_logits = torch.cat(all_test_logits).cpu()
    all_test_preds = all_test_logits
    all_test_labels = torch.cat(all_test_labels).cpu()

    # compute & store loss
    with torch.no_grad():
        loss = CoxLoss(all_test_labels[:, 0], all_test_labels[:, 1], all_test_preds, device='cpu').item()
        cindex_test = concordance_index(all_test_labels[:, 0], -all_test_preds, all_test_labels[:, 1])
        pvalue_test = cox_log_rank(all_test_preds, all_test_labels[:, 1], all_test_labels[:, 0])
    print('Test loss {}'.format(loss))
    print('Test c-index {}'.format(cindex_test))
    print('Test p-value {}'.format(pvalue_test))

