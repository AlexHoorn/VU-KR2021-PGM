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
    n_nodes = list(range(5, 30, 5)) * 20
    tasks = [random_task(n, 2) for n in n_nodes]
    # Randomize order so not all big networks are executed at the same time
    random.shuffle(tasks)

    # Useful to restart an experiment in case of a crash
    with open("tasks_dump.pkl", "wb") as f:
        dump(tasks, f)

    # with open("tasks_dump.pkl", "rb") as f:
    #     tasks = load(f)

    # MARGINALS EXPERIMENT
    run_experiment(experiment_marginal, tasks, n_threads=8)
    # JOINT EXPERIMENT
    run_experiment(experiment_joint, tasks, n_threads=8)


def experiment_marginal(
    task: Tuple[BNReasoner, List[str], Dict[str, bool]]
) -> Dict[str, Any]:
    bnreasoner = deepcopy(task[0])
    Q = task[1]
    E = task[2]

    info = get_graph_info(bnreasoner.bn.structure)
    info["n_query_vars"] = len(Q)
    info["n_query_evidence"] = len(E)

    # Marginal experiment
    info["experiment"] = "marginal"

    runtime = perf_counter()

    try:
        # Experiment here
        bnreasoner.pruning(Q, pd.Series(E))
        bnreasoner.marginal_distribution(Q, E)
    except Exception as e:
        info["exception"] = str(e)

    info["runtime"] = perf_counter() - runtime

    return info


def experiment_joint(
    task: Tuple[BNReasoner, List[str], Dict[str, bool]]
) -> Dict[str, Any]:
    # Deepcopy because pruning is inplace
    bnreasoner = deepcopy(task[0])
    Q = task[1]
    E = task[2]

    info = get_graph_info(bnreasoner.bn.structure)
    info["n_query_vars"] = len(Q)
    info["n_query_evidence"] = len(E)

    # Joint experiment
    info["experiment"] = "join"

    runtime = perf_counter()

    try:
        # Experiment here
        bnreasoner.pruning(Q, pd.Series(E))
        bnreasoner.joint_probability(Q, E)
    except Exception as e:
        info["exception"] = str(e)

    info["runtime"] = perf_counter() - runtime

    return info


if __name__ == "__main__":
    main()
