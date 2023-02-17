"""CellGraph Dataset loader."""
import os
import h5py
import torch.utils.data
import numpy as np
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from glob import glob 
import dgl 

from histocartography.utils import set_graph_on_cuda


IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
COLLATE_FN = {
    'DGLHeteroGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'int': lambda x: torch.LongTensor(x).to(DEVICE)
}


def h5_to_tensor(h5_path):
    h5_object = h5py.File(h5_path, 'r')
    out = torch.from_numpy(np.array(h5_object['assignment_matrix']))
    return out


class CellGraphDataset(Dataset):
    """CellGraph dataset."""

    def __init__(
            self,
            cg_path: str = None,
            load_in_ram: bool = False,
    ):
        """
        CellGraph dataset constructor.

        Args:
            cg_path (str, optional): Cell Graph path to a given split (eg, cell_graphs/test/). Defaults to None.
            tg_path (str, optional): Tissue Graph path. Defaults to None.
            assign_mat_path (str, optional): Assignment matrices path. Defaults to None.
            load_in_ram (bool, optional): Loading data in RAM. Defaults to False.
        """
        super(CellGraphDataset, self).__init__()
        self.cg_path = cg_path
        self.load_in_ram = load_in_ram

        if cg_path is not None:
            self._load_cg()

    def _load_cg(self):
        """
        Load cell graphs
        """
        self.cg_fnames = []
        self.cg_fnames = self.cg_fnames + glob(os.path.join(self.cg_path, '*.bin'))
        # patients = os.listdir(self.cg_path)
        # for patient in patients:
        #     self.cg_fnames = self.cg_fnames + glob(os.path.join(self.cg_path, patient, '*.bin'))

        # self.cg_fnames.sort()
        self.num_cg = len(self.cg_fnames)
        if self.load_in_ram:
            cell_graphs, label_dicts = [load_graphs(fname) for fname in self.cg_fnames]
            self.cell_graphs = [entry[0][0] for entry in cell_graphs]
            self.cell_graph_labels = label_dicts['CoxLabel']

    def __getitem__(self, index):
        """
        Get an example.
        Args:
            index (int): index of the example.
        """

        # CG-GNN configuration
        if self.load_in_ram:
            cg = self.cell_graphs[index]
            label = self.cell_graph_labels[index]
        else:
            cg, label = load_graphs(self.cg_fnames[index])
            label = label['CoxLabel'].to(DEVICE)
            cg = cg[0].to(DEVICE)
            cg.ndata['feat'] = cg.ndata['feat'].float()
            # cg = set_graph_on_cuda(cg) if IS_CUDA else cg
            return cg, label

    def __len__(self):
        """Return the number of samples in the CellGraph dataset."""
        if hasattr(self, 'num_cg'):
            return self.num_cg


def collate(batch):
    """
    Collate a batch.
    Args:
        batch (torch.tensor): a batch of examples.
    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """
    def collate_fn(batch, id, type):
        return COLLATE_FN[type]([example[id] for example in batch])

    # collate the data
    num_modalities = len(batch[0])  # should 2 if CG or TG processing or 4 if HACT
    batch = tuple([collate_fn(batch, mod_id, type(batch[0][mod_id]).__name__)
                  for mod_id in range(num_modalities)])
    return batch


def make_data_loader(
        batch_size,
        shuffle=True,
        num_workers=0,
        **kwargs
    ):
    """
    Create a CellGraph data loader.
    """
    dataset = CellGraphDataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate
    )
    return dataloader
