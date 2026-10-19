"""Per-slice ARI evaluation helpers for DLPFC-style AnnData objects."""

from pathlib import Path

from sklearn.metrics import (
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)


def valid_label_mask(labels, missing_label="unknown"):
    """Return a mask excluding nulls and normalized missing-label strings."""
    normalized_missing_label = missing_label.strip().casefold()
    object_labels = labels.astype("object")
    is_missing_label = object_labels.map(
        lambda label: (
            isinstance(label, str)
            and label.strip().casefold() == normalized_missing_label
        )
    )
    return labels.notna() & ~is_missing_label


def map_cluster_labels(predictions, mapping):
    """Map clusters to uniform string labels with explicit unmapped values."""
    return [
        (
            str(mapping[prediction])
            if prediction in mapping
            else f"unmapped_cluster_{prediction}"
        )
        for prediction in predictions
    ]


def evaluate_slice_ari(
    adata,
    section_ids,
    label_key,
    prediction_key,
    batch_key,
    ari_metric,
):
    """Return callback-ready per-section ARI, ACC, and mean values.

    ACC is computed as (NMI + homogeneity + completeness) / 3 per slice,
    matching the ``get_metrics`` convention used in the DLPFC pipeline.
    """
    section_ids = list(section_ids)
    if not section_ids:
        raise ValueError("section_ids must not be empty")

    slice_ari = {}
    slice_acc = {}
    for section_id in section_ids:
        labeled_mask = (adata.obs[batch_key] == section_id) & valid_label_mask(
            adata.obs[label_key]
        )
        section_adata = adata[labeled_mask]
        if len(section_adata.obs) == 0:
            raise ValueError(f"section {section_id!r} has no labeled spots")

        gt = section_adata.obs[label_key]
        pred = section_adata.obs[prediction_key]

        slice_ari[section_id] = float(ari_metric(gt, pred))

        nmi = normalized_mutual_info_score(gt, pred)
        hom = homogeneity_score(gt, pred)
        com = completeness_score(gt, pred)
        slice_acc[section_id] = float((nmi + hom + com) / 3.0)

    mean_ari = float(sum(slice_ari.values()) / len(slice_ari))
    mean_acc = float(sum(slice_acc.values()) / len(slice_acc))
    return {
        "score": mean_ari,
        "mean_ari": mean_ari,
        "mean_acc": mean_acc,
        "slice_ari": slice_ari,
        "slice_acc": slice_acc,
    }


def _format_slice_values(slice_dict, suffix="ARI"):
    return ", ".join(
        f"{section_id}_{suffix}={float(value):.4f}"
        for section_id, value in slice_dict.items()
    )


def format_ari_summary(history, best_evaluation):
    """Format complete periodic ARI/ACC history and the restored best result."""
    lines = ["Periodic ARI history:"]
    for result in history:
        slice_values = _format_slice_values(result["slice_ari"], suffix="ARI")
        lines.append(
            f"[Epoch {result['epoch']}] {slice_values}, "
            f"mean_ARI={float(result['mean_ari']):.4f}"
        )

    lines.append("Best periodic ARI:")
    if best_evaluation is None:
        lines.append("best_evaluation=None")
    else:
        slice_ari_values = _format_slice_values(
            best_evaluation["slice_ari"], suffix="ARI"
        )
        lines.append(
            f"best_epoch={best_evaluation['epoch']}, "
            f"best_mean_ARI={float(best_evaluation['mean_ari']):.4f}, "
            f"{slice_ari_values}"
        )
        # Also print ACC if available in the evaluation result
        if "slice_acc" in best_evaluation:
            slice_acc_values = _format_slice_values(
                best_evaluation["slice_acc"], suffix="ACC"
            )
            lines.append(
                f"best_epoch={best_evaluation['epoch']}, "
                f"best_mean_ACC={float(best_evaluation['mean_acc']):.4f}, "
                f"{slice_acc_values}"
            )
    return lines


def print_training_report(
    ari_list,
    acc_list,
    dis_list,
    nmi_list,
    history,
    best_evaluation,
    printer=print,
):
    """Print final metrics followed by complete periodic ARI details."""
    printer(f"ARI: {ari_list}")
    printer(f"ACC: {acc_list}")
    printer(f"DIS: {dis_list}")
    printer(f"NMI: {nmi_list}")
    for line in format_ari_summary(history, best_evaluation):
        printer(line)


def report_then_run(
    action,
    *,
    ari_list,
    acc_list,
    dis_list,
    nmi_list,
    history,
    best_evaluation,
    printer=print,
):
    """Print the complete report before running and returning an action."""
    print_training_report(
        ari_list,
        acc_list,
        dis_list,
        nmi_list,
        history,
        best_evaluation,
        printer=printer,
    )
    return action()


def build_figure_path(base_dir, timestamp):
    """Create the project output directory and return a timestamped PNG path."""
    output_dir = Path(base_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"figure_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
