import os
import sys

import numpy as np

# Prefer an explicitly configured R; otherwise use the active environment's R
# when present, or let rpy2 discover the system installation.
_environment_r = os.path.join(sys.prefix, "lib", "R")
if os.path.isdir(_environment_r):
    os.environ.setdefault("R_HOME", _environment_r)


def fix_seed(seed=2024):
    import random
    import torch
    from torch.backends import cudnn

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def mclust_R(
    adata,
    num_cluster,
    modelNames="EEE",
    used_obsm="model_pred",
    key_added_pred="mclust",
    random_seed=2024,
):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects

    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r["set.seed"]
    r_random_seed(random_seed)
    rmclust = robjects.r["Mclust"]
    res = rmclust(
        rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames
    )
    mclust_res = np.array(res[-2])
    adata.obs[key_added_pred] = mclust_res
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype("int")
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype("category")
    return adata
