import os
import random
from copy import deepcopy
from multiprocessing import Pool
from pickle import dump, load
from time import perf_counter
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from benchmark_utils import get_graph_info, random_task, run_experiment
from BNReasoner import BNReasoner

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():
    n_nodes = list(range(5, 515, 10)) * 10
    tasks = []
    with Pool() as pool:
        for t in tqdm(
            pool.imap_unordered(random_task, n_nodes),
            total=len(n_nodes),
            desc="Generating tasks",
        ):
            tasks.append(t)

    # Randomize order so not all big networks are executed at the same time
    random.shuffle(tasks)

    # Useful to restart an experiment in case of a crash
    with open("tasks_dump.pkl", "wb") as f:
        dump(tasks, f)

    # with open("tasks_dump.pkl", "rb") as f:
    #     tasks = load(f)

    # Orderings
    experiments = [
        experiment_order_random,
        experiment_order_mindeg,
        experiment_order_minfill,
    ]
    for exp in experiments:
        run_experiment(exp, tasks)

    df = pd.concat([pd.read_csv(f"{exp.__name__}_results.csv") for exp in experiments])
    df.to_csv("ordering_results.csv", index=False)


def experiment_order_random(
    task: Tuple[BNReasoner, List[str], Dict[str, bool]]
) -> Dict[str, Any]:
    bnreasoner = deepcopy(task[0])
    info = get_graph_info(bnreasoner.bn.structure)

    # Marginal experiment
    info["experiment"] = "order_random"

    runtime = perf_counter()

    try:
        # Experiment here
        bnreasoner.order(heuristic="random")
    except Exception as e:
        info["exception"] = str(e)

    info["runtime"] = perf_counter() - runtime

    return info


def experiment_order_mindeg(
    task: Tuple[BNReasoner, List[str], Dict[str, bool]]
) -> Dict[str, Any]:
    bnreasoner = deepcopy(task[0])
    info = get_graph_info(bnreasoner.bn.structure)

    # Marginal experiment
    info["experiment"] = "order_mindeg"

    runtime = perf_counter()

    try:
        # Experiment here
        bnreasoner.order(heuristic="mindeg")
    except Exception as e:
        info["exception"] = str(e)

    info["runtime"] = perf_counter() - runtime

    return info


def experiment_order_minfill(
    task: Tuple[BNReasoner, List[str], Dict[str, bool]]
) -> Dict[str, Any]:
    bnreasoner = deepcopy(task[0])
    info = get_graph_info(bnreasoner.bn.structure)

    # Marginal experiment
    info["experiment"] = "order_minfill"

    runtime = perf_counter()

    try:
        # Experiment here
        bnreasoner.order(heuristic="minfill")
    except Exception as e:
        info["exception"] = str(e)

    info["runtime"] = perf_counter() - runtime

    return info


if __name__ == "__main__":
    main()
