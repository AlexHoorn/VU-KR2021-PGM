import os
import random
from copy import deepcopy
from pickle import dump, load
from time import perf_counter
from typing import Any, Dict, List, Tuple

import pandas as pd

from benchmark_utils import get_graph_info, random_task, run_experiment
from BNReasoner import BNReasoner

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():
    n_nodes = list(range(5, 30, 5)) * 10
    tasks = [random_task(n, 2) for n in n_nodes]
    # Randomize order so not all big networks are executed at the same time
    random.shuffle(tasks)

    # Useful to restart an experiment in case of a crash
    with open("tasks_dump.pkl", "wb") as f:
        dump(tasks, f)

    # with open("tasks_dump.pkl", "rb") as f:
    #     tasks = load(f)

    # MPEs
    run_experiment(experiment_mpe_random, tasks)
    run_experiment(experiment_mpe_mindeg, tasks)
    run_experiment(experiment_mpe_minfill, tasks)
    # MAPs
    run_experiment(experiment_map_random, tasks)
    run_experiment(experiment_map_mindeg, tasks)
    run_experiment(experiment_map_minfill, tasks)


def experiment_map_random(
    task: Tuple[BNReasoner, List[str], Dict[str, bool]]
) -> Dict[str, Any]:
    bnreasoner = deepcopy(task[0])
    Q = task[1]
    E = pd.Series(task[2])

    info = get_graph_info(bnreasoner.bn.structure)
    info["n_query_vars"] = len(Q)
    info["n_query_evidence"] = len(E)

    # Marginal experiment
    info["experiment"] = "map_random"

    runtime = perf_counter()

    try:
        # Experiment here
        bnreasoner.map_mpe_estimation(Q=Q, E=E, heuristic="random")
    except Exception as e:
        info["exception"] = str(e)

    info["runtime"] = perf_counter() - runtime

    return info


def experiment_map_mindeg(
    task: Tuple[BNReasoner, List[str], Dict[str, bool]]
) -> Dict[str, Any]:
    bnreasoner = deepcopy(task[0])
    Q = task[1]
    E = pd.Series(task[2])

    info = get_graph_info(bnreasoner.bn.structure)
    info["n_query_vars"] = len(Q)
    info["n_query_evidence"] = len(E)

    # Marginal experiment
    info["experiment"] = "map_mindeg"

    runtime = perf_counter()

    try:
        # Experiment here
        bnreasoner.map_mpe_estimation(Q=Q, E=E, heuristic="mindeg")
    except Exception as e:
        info["exception"] = str(e)

    info["runtime"] = perf_counter() - runtime

    return info


def experiment_map_minfill(
    task: Tuple[BNReasoner, List[str], Dict[str, bool]]
) -> Dict[str, Any]:
    bnreasoner = deepcopy(task[0])
    Q = task[1]
    E = pd.Series(task[2])

    info = get_graph_info(bnreasoner.bn.structure)
    info["n_query_vars"] = len(Q)
    info["n_query_evidence"] = len(E)

    # Marginal experiment
    info["experiment"] = "map_minfill"

    runtime = perf_counter()

    try:
        # Experiment here
        bnreasoner.map_mpe_estimation(Q=Q, E=E, heuristic="minfill")
    except Exception as e:
        info["exception"] = str(e)

    info["runtime"] = perf_counter() - runtime

    return info


def experiment_mpe_random(
    task: Tuple[BNReasoner, List[str], Dict[str, bool]]
) -> Dict[str, Any]:
    bnreasoner = deepcopy(task[0])
    Q = None
    E = pd.Series(task[2])

    info = get_graph_info(bnreasoner.bn.structure)
    info["n_query_vars"] = 0
    info["n_query_evidence"] = len(E)

    # Marginal experiment
    info["experiment"] = "mpe_random"

    runtime = perf_counter()

    try:
        # Experiment here
        bnreasoner.map_mpe_estimation(Q=Q, E=E, heuristic="random")
    except Exception as e:
        info["exception"] = str(e)

    info["runtime"] = perf_counter() - runtime

    return info


def experiment_mpe_mindeg(
    task: Tuple[BNReasoner, List[str], Dict[str, bool]]
) -> Dict[str, Any]:
    bnreasoner = deepcopy(task[0])
    Q = None
    E = pd.Series(task[2])

    info = get_graph_info(bnreasoner.bn.structure)
    info["n_query_vars"] = 0
    info["n_query_evidence"] = len(E)

    # Marginal experiment
    info["experiment"] = "mpe_mindeg"

    runtime = perf_counter()

    try:
        # Experiment here
        bnreasoner.map_mpe_estimation(Q=Q, E=E, heuristic="mindeg")
    except Exception as e:
        info["exception"] = str(e)

    info["runtime"] = perf_counter() - runtime

    return info


def experiment_mpe_minfill(
    task: Tuple[BNReasoner, List[str], Dict[str, bool]]
) -> Dict[str, Any]:
    bnreasoner = deepcopy(task[0])
    Q = None
    E = pd.Series(task[2])

    info = get_graph_info(bnreasoner.bn.structure)
    info["n_query_vars"] = 0
    info["n_query_evidence"] = len(E)

    # Marginal experiment
    info["experiment"] = "mpe_minfill"

    runtime = perf_counter()

    try:
        # Experiment here
        bnreasoner.map_mpe_estimation(Q=Q, E=E, heuristic="minfill")
    except Exception as e:
        info["exception"] = str(e)

    info["runtime"] = perf_counter() - runtime

    return info


if __name__ == "__main__":
    main()
