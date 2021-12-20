import os
import random
from concurrent.futures import TimeoutError
from functools import cache
from multiprocessing import cpu_count
from multiprocessing.spawn import freeze_support
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from networkx import DiGraph
from pebble import ProcessPool
from pebble.common import ProcessExpired
from tqdm import tqdm

from BayesNet import BayesNet
from BNReasoner import BNReasoner

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def run_experiment(
    experiment: Callable,
    tasks: Tuple[BNReasoner, List[str], Dict[str, bool]],
    n_threads: int = cpu_count(),
) -> None:
    results = []
    timeouts = 0

    with ProcessPool(n_threads, max_tasks=1) as pool:
        with tqdm(total=len(tasks), desc=f"Executing {experiment.__name__}") as pbar:
            future = pool.map(experiment, tasks, timeout=120)
            iterator = future.result()

            while True:
                try:
                    results.append(next(iterator))
                except StopIteration:
                    break
                except TimeoutError:
                    timeouts += 1
                    pbar.set_description(
                        f"Executing {experiment.__name__} ({timeouts=})"
                    )
                except ProcessExpired:
                    tqdm.write("Process expired")
                except Exception as error:
                    tqdm.write("Function raised %s" % error)
                    tqdm.write(error.traceback)
                finally:
                    pbar.update()

    pd.DataFrame(results).to_csv(f"{experiment.__name__}_results.csv", index=False)


@cache
def load_reasoner(file: str) -> BNReasoner:
    bnet = BayesNet()
    bnet.load_from_bifxml(file)
    bnreasoner = BNReasoner(bnet)

    return bnreasoner


def random_task(n_nodes, mean_edges=2) -> Tuple[BNReasoner, List[str], Dict[str, bool]]:
    bnet = BayesNet()
    bnet.load_random(n_nodes, mean_edges)
    bnreasoner = BNReasoner(bnet)

    G = bnreasoner.bn.structure
    Q = random_variables(G)
    E = random_evidence(G)

    return bnreasoner, Q, E


def random_variables(G: DiGraph, p: float = 0.1, max_vars=5) -> List[str]:
    nodes = list(G.nodes)
    random_nodes = set()

    n = max(int(round(len(nodes) * p)), 1)
    if max_vars is not None:
        n = min(max_vars, n)

    random_nodes = {random.choice(list(nodes)) for _ in range(n)}
    return list(random_nodes)


def random_evidence(G: DiGraph, p: float = 0.8) -> Dict[str, bool]:
    nodes = random_variables(G, p=p, max_vars=None)
    return {node: random.choice([True, False]) for node in nodes}


def get_graph_info(G: DiGraph) -> Dict[str, Any]:
    n_predecessors = [len(list(G.predecessors(node))) for node in G.nodes]
    n_successors = [len(list(G.successors(node))) for node in G.nodes]
    n_edges = [p + s for p, s in zip(n_predecessors, n_successors)]

    info = dict(
        nodes=len(G.nodes),
        roots=len([p for p in n_predecessors if p == 0]),
        leaves=len([s for s in n_successors if s == 0]),
        edges=len(G.edges),
        mean_predecessors=sum(n_predecessors) / len(n_predecessors),
        max_predecessors=max(n_predecessors),
        min_predecessors=min(n_predecessors),
        mean_successors=sum(n_successors) / len(n_successors),
        max_successors=max(n_edges),
        min_successors=min(n_successors),
        mean_edges=sum(n_edges) / len(n_edges),
        max_edges=max(n_edges),
        min_edges=min(n_edges),
    )

    return info
