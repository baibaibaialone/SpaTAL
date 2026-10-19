import copy
import random
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm

from .evaluation import EvaluationCheckpointManager, should_evaluate
from .learner import ATT_learner
from .model import SpaTALModel
from .preprocess import Transfer_pytorch_Data
from .utils import fix_seed, mclust_R


def symmetrize(adj):
    return (adj + adj.T) / 2


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(
        feat_node, size=int(feat_node * mask_rate), replace=False
    )
    mask[:, samples] = 1
    return mask


def _dense_adjacency(data):
    adjacency = torch.zeros(
        (data.num_nodes, data.num_nodes),
        dtype=torch.float32,
        device=data.edge_index.device,
    )
    adjacency[data.edge_index[0], data.edge_index[1]] = 1.0
    return adjacency


@contextmanager
def _preserve_random_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def loss_gcl(model, graph_learner, features, anchor_adj):
    mask_v1 = get_feat_mask(features, 0.2)
    features_v1 = features * (1 - mask_v1.to(device=features.device))
    z1, _ = model(features_v1, anchor_adj)

    mask_v2 = get_feat_mask(features, 0.2)
    features_v2 = features * (1 - mask_v2.to(device=features.device))
    learned_adj = normalize(symmetrize(graph_learner(features)))
    learned_adj = learned_adj.to(device=features.device)
    z2, _ = model(features_v2, learned_adj)

    return model.calc_loss(z1, z2)


EOS = 1e-10


def normalize(adj):
    inv_sqrt_degree = 1.0 / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
    return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]


class SpaTAL:
    def __init__(
        self,
        adata,
        tissue_name="DLPFC",
        num_clusters=7,
        device=torch.device("cuda"),
        learning_rate=0.001,
        weight_decay=2e-4,
        max_epoch=1650,
        gradient_clipping=5,
        random_seed=2023,
        nps=30,
    ):

        self.__adata = adata.copy()
        self.__tissue_name = tissue_name
        self.__device = device
        self.__learning_rate = learning_rate
        self.__weight_decay = weight_decay
        self.__max_epoch = max_epoch
        self.__gradient_clipping = gradient_clipping
        self.__nps = nps
        self.evaluation_history = []
        self.best_evaluation = None

        fix_seed(random_seed)

        if "feat" not in self.__adata.obsm:
            raise ValueError("Preprocessed input must include adata.obsm['feat'].")
        if "Spatial_Net" not in self.__adata.uns:
            raise ValueError(
                "Preprocessed input must include adata.uns['Spatial_Net']."
            )

        self.num_clusters = num_clusters
        print(self.__adata.obsm["feat"].shape)

    def train(self, eval_interval=None, eval_method="mclust", eval_callback=None):
        data = Transfer_pytorch_Data(self.__adata).to(self.__device)

        anchor_adj = normalize(_dense_adjacency(data))

        k = 15
        print("k:", k)
        nlayers = 4
        print("nlayers:", nlayers)

        self.k = k
        self.nlayers = nlayers

        self.graph_learner = ATT_learner(nlayers, data.x.shape[1], k).to(self.__device)

        self.model = SpaTALModel().to(self.__device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.__learning_rate,
            weight_decay=self.__weight_decay,
        )
        self.optimizer_learner = torch.optim.Adam(
            self.graph_learner.parameters(),
            self.__learning_rate,
            weight_decay=self.__weight_decay,
        )
        manager = EvaluationCheckpointManager()

        epoch_iter = tqdm(
            range(0, self.__max_epoch), desc="Epochs", unit="ep", leave=True
        )
        for epoch in epoch_iter:
            self.model.train()
            self.graph_learner.train()

            loss = loss_gcl(self.model, self.graph_learner, data.x, anchor_adj)

            self.optimizer.zero_grad()
            self.optimizer_learner.zero_grad()

            loss.backward()
            gradient_clipping = self.__gradient_clipping
            if gradient_clipping > 1:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clipping
                )

            self.optimizer.step()
            self.optimizer_learner.step()

            completed_epoch = epoch + 1
            if (
                eval_callback is not None
                and eval_interval is not None
                and should_evaluate(
                    completed_epoch,
                    self.__max_epoch,
                    eval_interval,
                )
            ):
                with _preserve_random_state():
                    self.process(method=eval_method)
                    result = eval_callback(completed_epoch, self.__adata)
                    manager.record(
                        epoch=completed_epoch,
                        result=result,
                        model=self.model,
                        graph_learner=self.graph_learner,
                        adata=self.__adata,
                    )

            epoch_iter.set_postfix(loss=f"{loss.item():.4f}")
            tqdm.write(
                f"Dataset_Name:{self.__tissue_name}, Ep {epoch}: train loss:{loss.item():.4f}"
            )

        self.evaluation_history = copy.deepcopy(manager.history)
        self.best_evaluation = copy.deepcopy(manager.best_result)
        restored_adata = manager.restore(self.model, self.graph_learner)
        if restored_adata is not None:
            self.__adata = restored_adata
        return copy.deepcopy(self.evaluation_history)

    def process(self, method="mclust"):
        if method != "mclust":
            raise ValueError("Only mclust evaluation is supported.")
        data = Transfer_pytorch_Data(self.__adata).to(self.__device)

        with torch.no_grad():
            self.model.eval()
            self.graph_learner.eval()

            anchor_adj = normalize(_dense_adjacency(data))

            _, embedding = self.model(data.x, anchor_adj)

            embedding = embedding.to("cpu").detach().numpy()

            if embedding.shape[-1] > 64:
                from sklearn.decomposition import PCA

                pca = PCA(n_components=self.__nps)
                embedding = pca.fit_transform(embedding)
            self.__adata.obsm["eval_pred"] = embedding

            mclust_R(
                self.__adata,
                num_cluster=self.num_clusters,
                used_obsm="eval_pred",
                key_added_pred=method,
            )

    def get_adata(self):
        return self.__adata
