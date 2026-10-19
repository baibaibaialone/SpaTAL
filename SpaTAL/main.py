from datetime import datetime
import random

import scipy.sparse
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)
random.seed(42)

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    pairwise_distances,
)
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
import anndata as ad
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import os
from pathlib import Path
import torch
from spatal import SpaTAL
from spatal.config import Config
import scanpy as sc
from sklearn import metrics
from sklearn.decomposition import PCA
from spatal.metrics import (
    build_figure_path,
    evaluate_slice_ari,
    map_cluster_labels,
    report_then_run,
    valid_label_mask,
)


def _compute_CHAOS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1

    return np.sum(dist_val) / len(clusterlabel)


def fx_1NN(i, location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    return np.min(dist_array)


def fx_kNN(i, location_in, k, cluster_in):
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)

    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind] != cluster_in[i]) > (k / 2):
        return 1
    else:
        return 0


def _compute_PAS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [
        fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel)
        for i in range(matched_location.shape[0])
    ]
    return np.sum(results) / len(clusterlabel)


def compute_ARI(adata, gt_key, pred_key):
    return adjusted_rand_score(adata.obs[gt_key], adata.obs[pred_key])


def compute_NMI(adata, gt_key, pred_key):
    return normalized_mutual_info_score(adata.obs[gt_key], adata.obs[pred_key])


def compute_CHAOS(adata, pred_key, spatial_key="spatial"):
    return _compute_CHAOS(adata.obs[pred_key], adata.obsm[spatial_key])


def compute_PAS(adata, pred_key, spatial_key="spatial"):
    return _compute_PAS(adata.obs[pred_key], adata.obsm[spatial_key])


def compute_HOM(adata, gt_key, pred_key):
    return homogeneity_score(adata.obs[gt_key], adata.obs[pred_key])


def compute_COM(adata, gt_key, pred_key):
    return completeness_score(adata.obs[gt_key], adata.obs[pred_key])


def get_metrics(adata, label_key, prediction_key):
    adata = adata[valid_label_mask(adata.obs[label_key])]
    ARI = compute_ARI(adata, label_key, prediction_key)
    NMI = compute_NMI(adata, label_key, prediction_key)
    HOM = compute_HOM(adata, label_key, prediction_key)
    COM = compute_COM(adata, label_key, prediction_key)
    CHAOS = compute_CHAOS(adata, prediction_key)
    PAS = compute_PAS(adata, prediction_key)
    return ARI, (NMI + HOM + COM) / 3, (CHAOS + PAS) / 2, NMI


def align_spots(adata_st_list_input, tol=0.01):
    adata_st_list = adata_st_list_input.copy()
    print("Using the Iterative Closest Point algorithm for alignemnt.")
    print("Detecting edges...")
    point_cloud_list = []
    for adata in adata_st_list:
        if "in_tissue" in adata.obs.columns:
            adata = adata[adata.obs["in_tissue"] == 1]
        loc_x = adata.obs.loc[:, ["array_row"]]
        loc_x = np.array(loc_x) * np.sqrt(3)
        loc_y = adata.obs.loc[:, ["array_col"]]
        loc_y = np.array(loc_y)
        loc = np.concatenate((loc_x, loc_y), axis=1)
        pairwise_loc_distsq = np.sum(
            (loc.reshape([1, -1, 2]) - loc.reshape([-1, 1, 2])) ** 2, axis=2
        )
        n_neighbors = np.sum(pairwise_loc_distsq < 5, axis=1) - 1
        edge = ((n_neighbors > 1) & (n_neighbors < 5)).astype(np.float32)
        point_cloud_list.append(adata.obsm["spatial"][edge == 1].copy())
    print("Aligning edges...")
    trans_list = []
    adata_st_list[0].obsm["spatial_aligned"] = adata_st_list[0].obsm["spatial"].copy()
    for i in range(len(adata_st_list) - 1):
        T, _, _ = icp(point_cloud_list[i + 1], point_cloud_list[i], tolerance=tol)
        trans_list.append(T)
    for i in range(len(adata_st_list) - 1):
        point_cloud_align = adata_st_list[i + 1].obsm["spatial"].copy()
        for T in trans_list[: i + 1][::-1]:
            point_cloud_align = transform(point_cloud_align, T)
        adata_st_list[i + 1].obsm["spatial_aligned"] = point_cloud_align
    return adata_st_list


# ICP implementation: https://github.com/ClayFlannigan/icp
def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


def transform(point_cloud, T):
    point_cloud_align = np.ones((point_cloud.shape[0], 3))
    point_cloud_align[:, 0:2] = np.copy(point_cloud)
    point_cloud_align = np.dot(T, point_cloud_align.T).T
    return point_cloud_align[:, :2]


def preprocess(adata_st_list, section_ids, k_cutoff, slice_dist_micron, c2c_dist=100):
    adata_st = ad.concat(adata_st_list, label="slice_name", keys=section_ids)
    adata_st.obs["Ground Truth"] = adata_st.obs["Ground Truth"].astype("category")
    adata_st.obs["batch_name"] = adata_st.obs["slice_name"].astype("category")
    print("Start building a graph...")
    adata_st_ref = adata_st_list[0].copy()
    loc_ref = np.array(adata_st_ref.obsm["spatial_aligned"])
    pair_dist_ref = pairwise_distances(loc_ref)
    min_dist_ref = np.sort(np.unique(pair_dist_ref), axis=None)[1]
    if len(slice_dist_micron) != len(adata_st_list) - 1:
        raise ValueError(
            "The length of 'slice_dist_micron' should be the number of adatas - 1 !"
        )
    else:
        loc_xy = pd.DataFrame(adata_st.obsm["spatial_aligned"]).values
        loc_z = np.zeros(adata_st.shape[0])
        dim = 0
        for i in range(len(slice_dist_micron)):
            dim += adata_st_list[i].shape[0]
            loc_z[dim:] += slice_dist_micron[i] * (min_dist_ref / c2c_dist)
        loc = np.concatenate([loc_xy, loc_z.reshape(-1, 1)], axis=1)
    loc = pd.DataFrame(loc)
    loc.index = adata_st.obs.index
    loc.columns = ["x", "y", "z"]
    nbrs = NearestNeighbors(n_neighbors=k_cutoff + 1).fit(loc)
    distances, indices = nbrs.kneighbors(loc)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(
            pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :]))
        )
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ["Cell1", "Cell2", "Distance"]
    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net["Distance"] > 0,]
    id_cell_trans = dict(zip(range(loc.shape[0]), np.array(loc.index)))
    Spatial_Net["Cell1"] = Spatial_Net["Cell1"].map(id_cell_trans)
    Spatial_Net["Cell2"] = Spatial_Net["Cell2"].map(id_cell_trans)
    print(
        "The graph contains %d edges, %d cells."
        % (Spatial_Net.shape[0], adata_st.n_obs)
    )
    print(
        "%.4f neighbors per cell on average." % (Spatial_Net.shape[0] / adata_st.n_obs)
    )
    adata_st.uns["Spatial_Net"] = Spatial_Net
    return adata_st


def load_adata(section_ids, k_cutoff, n_top_genes):
    Batch_list = []
    for section_id in section_ids:
        print(section_id)
        input_dir = os.path.join(
            os.environ.get("DLPFC_DATA_ROOT", "/tmp/baixu/DEEPST/data/DLPFC"),
            section_id,
        )
        adata = sc.read_visium(
            path=input_dir,
            count_file=section_id + "_filtered_feature_bc_matrix.h5",
            load_images=True,
        )
        adata.var_names_make_unique(join="++")
        Ann_df = pd.read_csv(
            os.path.join(input_dir, section_id + "_truth.txt"),
            sep="\t",
            header=None,
            index_col=0,
        )
        Ann_df.columns = ["Ground Truth"]
        adata.obs["Ground Truth"] = Ann_df.loc[adata.obs_names, "Ground Truth"].astype(
            "category"
        )
        adata.obs_names = [x + "_" + section_id for x in adata.obs_names]
        adata.var_names_make_unique()
        adata.layers["count"] = adata.X.toarray()
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1000000.0)
        sc.pp.highly_variable_genes(
            adata, flavor="seurat_v3", layer="count", n_top_genes=n_top_genes
        )
        adata = adata[:, adata.var["highly_variable"] == True]
        sc.pp.scale(adata)
        adata = adata[:, adata.var["highly_variable"]]
        Batch_list.append(adata)
    Batch_list = align_spots(Batch_list)
    adata_st = preprocess(
        Batch_list,
        section_ids=section_ids,
        k_cutoff=k_cutoff,
        slice_dist_micron=[10, 10, 10],
    )
    adata_X = PCA(n_components=200, random_state=1235).fit_transform(adata_st.X)
    adata_st.obsm["feat"] = adata_X
    return adata_st


def train_one(args, adata, section_ids, num_clusters):
    net = SpaTAL(
        adata,
        tissue_name="DLPFC",
        num_clusters=num_clusters,
        device=torch.device(
            os.environ.get(
                "DLPFC_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        ),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epoch=args.max_epoch,
        gradient_clipping=args.gradient_clipping,
        random_seed=args.seed,
        nps=args.nps,
    )
    method = "mclust"

    def evaluate_epoch(completed_epoch, evaluated_adata):
        result = evaluate_slice_ari(
            adata=evaluated_adata,
            section_ids=section_ids,
            label_key="Ground Truth",
            prediction_key=method,
            batch_key="batch_name",
            ari_metric=metrics.adjusted_rand_score,
        )
        slice_values = ", ".join(
            f"{section_id}_ARI={ari:.4f}"
            for section_id, ari in result["slice_ari"].items()
        )
        print(
            f"[Epoch {completed_epoch}] {slice_values}, "
            f"mean_ARI={result['mean_ari']:.4f}"
        )
        return result

    net.train(
        eval_interval=2000,
        eval_method=method,
        eval_callback=evaluate_epoch,
    )
    adata = net.get_adata()
    return adata, net.evaluation_history, net.best_evaluation, net


def main():
    args = Config()
    slices_list = ["151673", "151674", "151675", "151676"]
    num_clusters = 7
    adata = load_adata(slices_list, k_cutoff=args.k_cutoff, n_top_genes=args.top_genes)
    adata, evaluation_history, best_evaluation, net = train_one(
        args, adata, slices_list, num_clusters
    )
    PD_key = "mclust"
    GT_key = "Ground Truth"
    LT_key = "eval_pred"
    BT_key = "batch_name"
    valid_gt_mask = valid_label_mask(adata.obs[GT_key])
    pred_labels = adata.obs.loc[valid_gt_mask, PD_key].values
    gt_labels = adata.obs.loc[valid_gt_mask, GT_key].values
    pred_unique = np.unique(pred_labels)
    gt_unique = np.unique(gt_labels)
    n_pred = len(pred_unique)
    n_gt = len(gt_unique)
    cost_matrix = np.zeros((n_pred, n_gt))
    for i, p in enumerate(pred_unique):
        for j, g in enumerate(gt_unique):
            cost_matrix[i, j] = -np.sum((pred_labels == p) & (gt_labels == g))
    maps = {}
    if n_pred and n_gt:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        maps = {pred_unique[r]: gt_unique[c] for r, c in zip(row_ind, col_ind)}
    values = map_cluster_labels(adata.obs[PD_key], maps)
    data = {"cluster": values}
    df = pd.DataFrame(data, index=adata.obs[PD_key].index)
    adata.obs[PD_key] = df

    def render_umap():
        sc.pp.neighbors(adata, use_rep=LT_key, metric="cosine")
        sc.tl.umap(adata)
        umap_output_dir = Path(__file__).resolve().parent / "output"
        umap_output_dir.mkdir(parents=True, exist_ok=True)
        umap_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        umap_plots = (
            (GT_key, "ground_truth"),
            (PD_key, "mclust"),
            (BT_key, "batch_name"),
        )
        umap_paths = {}
        for color_key, filename_key in umap_plots:
            sc.pl.umap(adata, color=color_key, show=False, size=25, save=False)
            umap_path = umap_output_dir / f"umap_{filename_key}_{umap_timestamp}.png"
            umap_figure = plt.gcf()
            umap_figure.savefig(umap_path, dpi=150, bbox_inches="tight")
            plt.close(umap_figure)
            umap_paths[color_key] = umap_path
            print(f"UMAP图片已保存到: {umap_path}")
        return umap_paths

    Batch_list = []
    for section_id in slices_list:
        Batch_list.append(adata[adata.obs[BT_key] == section_id])
    ARI_LIST, ACC_LIST, DIS_LIST, NMI_LIST = ([], [], [], [])
    for idx in range(len(slices_list)):
        ARI, ACC, DIS, NMI = get_metrics(Batch_list[idx], GT_key, PD_key)
        ARI_LIST.append(round(ARI, 3))
        ACC_LIST.append(round(ACC, 3))
        DIS_LIST.append(round(DIS, 4))
        NMI_LIST.append(round(NMI, 3))
    umap_paths = report_then_run(
        render_umap,
        ari_list=ARI_LIST,
        acc_list=ACC_LIST,
        dis_list=DIS_LIST,
        nmi_list=NMI_LIST,
        history=evaluation_history,
        best_evaluation=best_evaluation,
    )
    gene_list = ["TMSB10"]
    gene_color_map = clr.LinearSegmentedColormap.from_list(
        "pink_green", ["#450756", "#451464", "#426189", "#2BD32B", "#F9F871"], N=256
    )
    gene_plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gene_plot_paths = []
    gene_output_dir = Path(__file__).resolve().parent / "output"
    gene_output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"axes.facecolor": "white", "figure.facecolor": "white"}):
        for section_id, adata_tmp in zip(slices_list, Batch_list):
            for gene_name in gene_list:
                if gene_name not in adata_tmp.var_names:
                    print(f"跳过gene图片: {gene_name}不在切片{section_id}的特征中")
                    continue
                gene_index = adata_tmp.var_names.get_loc(gene_name)
                gene_values = adata_tmp.X[:, gene_index]
                if scipy.sparse.issparse(gene_values):
                    gene_values = gene_values.toarray()
                expression_key = f"{gene_name}(raw)"
                adata_tmp.obs[expression_key] = np.asarray(gene_values).ravel()
                sc.pl.spatial(
                    adata_tmp,
                    img_key=None,
                    color=expression_key,
                    title=gene_name,
                    color_map=gene_color_map,
                    spot_size=200,
                    show=False,
                )
                gene_plot_path = (
                    gene_output_dir
                    / f"gene_{gene_name}_{section_id}_{gene_plot_timestamp}.png"
                )
                gene_figure = plt.gcf()
                gene_figure.patch.set_facecolor("white")
                for gene_axis in gene_figure.axes:
                    gene_axis.set_facecolor("white")
                gene_figure.savefig(
                    gene_plot_path,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                    transparent=False,
                )
                plt.close(gene_figure)
                gene_plot_paths.append(gene_plot_path)
                print(f"Gene图片已保存到: {gene_plot_path}")
    _log_timestamp = datetime.now()
    _base_dir = Path(__file__).resolve().parent
    _output_dir = _base_dir / "output"
    _output_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _output_dir / "results.log"
    _best_epoch = best_evaluation["epoch"] if best_evaluation else "N/A"
    _best_mean_ari = f"{best_evaluation['mean_ari']:.4f}" if best_evaluation else "N/A"
    _best_mean_acc = f"{best_evaluation['mean_acc']:.4f}" if best_evaluation else "N/A"
    _slice_ari_str = ", ".join(
        (f"{sid}_ARI={ARI_LIST[i]:.3f}" for i, sid in enumerate(slices_list))
    )
    _slice_acc_str = ", ".join(
        (f"{sid}_ACC={ACC_LIST[i]:.3f}" for i, sid in enumerate(slices_list))
    )
    with open(_log_path, "a") as log_f:
        log_f.write("=== SpaTAL Final Results ===\n")
        log_f.write(f"Timestamp: {_log_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"Best Epoch: {_best_epoch}\n")
        log_f.write(f"k_cutoff (graph): {args.k_cutoff}\n")
        log_f.write(f"top_genes: {args.top_genes}\n")
        for color_key, umap_path in umap_paths.items():
            log_f.write(f"UMAP figure ({color_key}): {umap_path}\n")
        for gene_plot_path in gene_plot_paths:
            log_f.write(f"Gene figure: {gene_plot_path}\n")
        log_f.write(f"k (graph learner): {net.k}\n")
        log_f.write(f"nlayers (graph learner): {net.nlayers}\n")
        log_f.write(f"num_clusters: {num_clusters}\n")
        log_f.write("\n--- Final ARI ---\n")
        log_f.write(f"mean_ARI: {_best_mean_ari}\n")
        log_f.write(f"{_slice_ari_str}\n")
        log_f.write("\n--- Final ACC ---\n")
        log_f.write(f"mean_ACC: {_best_mean_acc}\n")
        log_f.write(f"{_slice_acc_str}\n")
        log_f.write("\n--- All Metrics ---\n")
        log_f.write(f"ARI_LIST: {ARI_LIST}\n")
        log_f.write(f"ACC_LIST: {ACC_LIST}\n")
        log_f.write(f"DIS_LIST: {DIS_LIST}\n")
        log_f.write(f"NMI_LIST: {NMI_LIST}\n")
        log_f.write("\n")
    print(f"Result appended to: {_log_path}")
    spot_size = 200
    title_size = 12
    fig, ax = plt.subplots(
        1, 4, figsize=(10, 5), gridspec_kw={"wspace": 0.05, "hspace": 0.1}
    )
    for idx in range(len(slices_list)):
        _sc_ = sc.pl.spatial(
            Batch_list[idx],
            img_key=None,
            color=[PD_key],
            title=[""],
            legend_loc=None,
            legend_fontsize=12,
            show=False,
            ax=ax[idx],
            frameon=False,
            spot_size=spot_size,
        )
        _sc_[0].set_title("ARI=" + str(ARI_LIST[idx]), size=title_size)
    plt.show()
    figure_path = build_figure_path(Path(__file__).resolve().parent, datetime.now())
    plt.savefig(figure_path, dpi=150)
    print(f"图片已保存到: {figure_path}")


if __name__ == "__main__":
    main()
