#!/usr/bin/env python3
"""Aggregate one or more paper_bench CSVs into plotting-ready CSV files."""

import argparse
import collections
import csv
from pathlib import Path


FIELDNAMES = [
    "algorithm",
    "variant",
    "hosts",
    "label",
    "class",
    "dataset",
    "n_runs",
    "wall_s_mean",
    "kernel_s_mean",
    "load_s_mean",
    "cxl_load_s_mean",
    "cxl_store_s_mean",
    "wait_spins_mean",
    "cxl_load_s_perhost",
    "cxl_store_s_perhost",
    "compute_s",
]

ALGORITHM_ORDER = {
    name: index
    for index, name in enumerate(("bfs", "sort", "kmeans", "stencil", "minisat", "matmul", "bnb"))
}
VARIANT_ORDER = {
    "seq": 0,
    "threads": 1,
    "single": 0,
    "local": 0,
    "cxl": 1,
    "cxl_shared": 2,
    "pregel": 3,
    "cxl_partitioned": 4,
    "cxl_ipc": 3,
    "cxl_opt_dist": 3,
    "cxl_exchange": 3,
}
LABELS = {
    "seq": "seq",
    "threads": "threads",
    "single": "single",
    "local": "local",
    "cxl": "cxl",
    "cxl_shared": "shared",
    "pregel": "pregel",
    "cxl_partitioned": "part.",
    "cxl_ipc": "ipc",
    "cxl_opt_dist": "opt-dist",
    "cxl_exchange": "exch.",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="input benchmark CSV files (default: runs.csv)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="directory for aggregated.csv and per-algorithm .dat files",
    )
    return parser.parse_args()


def value(row, column):
    raw = row.get(column, "")
    return float(raw) if raw not in ("", None) else 0.0


def mean(group, column):
    return sum(value(row, column) for row in group) / len(group)


def label_for(algorithm, variant, hosts, run_class):
    if variant.startswith("ip_cxl_np"):
        return f"ip-cxl-{variant.removeprefix('ip_cxl_np')}"
    if variant.startswith("pregel_np"):
        return f"pregel-{variant.removeprefix('pregel_np')}"
    if variant.startswith("mpi_np"):
        ranks = variant.removeprefix("mpi_np")
        return f"mpi-{ranks}" if algorithm == "sort" else f"np{ranks}"
    label = LABELS.get(variant, variant)
    return f"{label}-{hosts}" if run_class == "cxl" else label


def sort_key(item):
    (algorithm, variant, hosts, _run_class, dataset), _group = item
    if variant.startswith("ip_cxl_np"):
        variant_order = 5 + int(variant.removeprefix("ip_cxl_np")) / 100
    elif variant.startswith("pregel_np"):
        variant_order = 3 + int(variant.removeprefix("pregel_np")) / 100
    elif variant.startswith("mpi_np"):
        variant_order = int(variant.removeprefix("mpi_np"))
    else:
        variant_order = VARIANT_ORDER.get(variant, 100)
    return (
        ALGORITHM_ORDER.get(algorithm, 100),
        variant_order,
        hosts,
        dataset,
        variant,
    )


def main():
    args = parse_args()
    runs_paths = args.runs or [Path("runs.csv")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for runs_path in runs_paths:
        with runs_path.open(newline="") as runs_file:
            rows.extend(csv.DictReader(runs_file))

    groups = collections.defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row["algorithm"],
            row["variant"],
            int(row["hosts"]),
            row["class"],
            row["dataset"],
        )
        groups[key].append(row)

    aggregated = []
    for (algorithm, variant, hosts, run_class, dataset), group in sorted(
        groups.items(), key=sort_key
    ):
        divisor = max(hosts, 1)
        wall = mean(group, "wall_s")
        kernel = mean(group, "kernel_s")
        load = mean(group, "load_s")
        cxl_load = mean(group, "cxl_load_s")
        cxl_store = mean(group, "cxl_store_s")
        wait_spins = mean(group, "wait_spins")
        cxl_load_perhost = cxl_load / divisor
        cxl_store_perhost = cxl_store / divisor
        compute = max(kernel - cxl_load_perhost - cxl_store_perhost, 0.0)

        aggregated.append(
            {
                "algorithm": algorithm,
                "variant": variant,
                "hosts": str(hosts),
                "label": label_for(algorithm, variant, hosts, run_class),
                "class": run_class,
                "dataset": dataset,
                "n_runs": str(len(group)),
                "wall_s_mean": f"{wall:.6f}",
                "kernel_s_mean": f"{kernel:.6f}",
                "load_s_mean": f"{load:.6f}",
                "cxl_load_s_mean": f"{cxl_load:.6f}",
                "cxl_store_s_mean": f"{cxl_store:.6f}",
                "wait_spins_mean": f"{wait_spins:.0f}",
                "cxl_load_s_perhost": f"{cxl_load_perhost:.6f}",
                "cxl_store_s_perhost": f"{cxl_store_perhost:.6f}",
                "compute_s": f"{compute:.6f}",
            }
        )

    aggregate_path = args.output_dir / "aggregated.csv"
    with aggregate_path.open("w", newline="") as aggregate_file:
        writer = csv.DictWriter(aggregate_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(aggregated)

    by_algorithm = collections.defaultdict(list)
    for row in aggregated:
        by_algorithm[row["algorithm"]].append(row)

    written = []
    for algorithm in ALGORITHM_ORDER:
        panel_path = args.output_dir / f"{algorithm}.dat"
        with panel_path.open("w", newline="") as panel_file:
            panel_file.write("variant,load,compute,cxlload,cxlstore\n")
            for row in by_algorithm.get(algorithm, []):
                panel_file.write(
                    f"{row['label']},{row['load_s_mean']},{row['compute_s']},"
                    f"{row['cxl_load_s_perhost']},{row['cxl_store_s_perhost']}\n"
                )
        written.append(panel_path.name)

    print(f"wrote: {aggregate_path.name}, {', '.join(written)}")


if __name__ == "__main__":
    main()
