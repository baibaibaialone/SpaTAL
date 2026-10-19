import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data


def Transfer_pytorch_Data(adata):
    """Convert the spatial graph and dense or sparse features to PyTorch data."""
    G_df = adata.uns["Spatial_Net"].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df["Cell1"] = G_df["Cell1"].map(cells_id_tran)
    G_df["Cell2"] = G_df["Cell2"].map(cells_id_tran)
    G = sp.coo_matrix(
        (np.ones(G_df.shape[0]), (G_df["Cell1"], G_df["Cell2"])),
        shape=(adata.n_obs, adata.n_obs),
    )
    G = G + sp.eye(G.shape[0])
    edgeList = np.nonzero(G)
    if type(adata.obsm["feat"]) == np.ndarray:
        data = Data(
            edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
            x=torch.FloatTensor(adata.obsm["feat"]),
        )
    else:
        data = Data(
            edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
            x=torch.FloatTensor(adata.obsm["feat"].todense()),
        )
    return data
