import random
from multiprocessing import Pool, freeze_support
from time import perf_counter
from typing import List

import pandas as pd
from networkx import DiGraph
from tqdm import tqdm
from BayesNet import BayesNet
from BNReasoner import BNReasoner
from functools import cache


def main():
    # Temporary (everything 20 times for statistic soundness)
    files = [
        r"testing\dog_problem.BIFXML",
        r"testing\lecture_example.BIFXML",
        r"testing\lecture_example2.BIFXML",
    ] * 20

    bnreasoners = [load_reasoner(file) for file in files]

    results = []
    with Pool() as pool:
        for result in tqdm(
            pool.imap_unordered(experiment, bnreasoners), total=len(bnreasoners)
        ):
            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("experiment_name_results.csv", index=False)


@cache
def load_reasoner(file: str):
    bnet = BayesNet()
    bnet.load_from_bifxml(file)
    bnreasoner = BNReasoner(bnet)

    return bnreasoner


def get_graph_info(G: DiGraph):
    n_predecessors = [len(list(G.predecessors(node))) for node in G.nodes]
    n_successors = [len(list(G.successors(node))) for node in G.nodes]
    n_edges = [p + s for p, s in zip(n_predecessors, n_successors)]

    info = dict(
        n_nodes=len(G.nodes),
        n_roots=len([p for p in n_predecessors if p == 0]),
        n_leaves=len([s for s in n_successors if s == 0]),
        n_edges=len(G.edges),
        mean_node_predecessors=sum(n_predecessors) / len(n_predecessors),
        max_node_predecessors=max(n_predecessors),
        min_node_predecessors=min(n_predecessors),
        mean_node_successors=sum(n_successors) / len(n_successors),
        max_node_successors=max(n_edges),
        min_node_successors=min(n_successors),
        mean_node_edges=sum(n_edges) / len(n_edges),
        max_node_edges=max(n_edges),
        min_node_edges=min(n_edges),
    )

    return info


def experiment(bnreasoner: BNReasoner):
    G = bnreasoner.bn.structure
    Q = random_variables(G)
    E = random_evidence(G)

    info = get_graph_info(G)
    info["n_vars"] = len(Q)
    info["n_evidence"] = len(E)

    runtime = perf_counter()

    # Measured experiment here
    # This is an example
    bnreasoner.pruning(Q, pd.Series(E))
    _ = bnreasoner.marginal_distribution(Q, E)

    info["runtime"] = perf_counter() - runtime

    return info


def random_variables(G: DiGraph) -> List[str]:
    nodes = set(G.nodes)
    n = random.randint(1, len(nodes))

    random_nodes = set()
    for _ in range(n):
        random_node = set([random.choice(list(nodes - random_nodes))])
        random_nodes = random_nodes | random_node

    return list(random_nodes)


def random_evidence(G: DiGraph):
    nodes = random_variables(G)
    return {node: random.choice([True, False]) for node in nodes}


if __name__ == "__main__":
    freeze_support()
    main()
