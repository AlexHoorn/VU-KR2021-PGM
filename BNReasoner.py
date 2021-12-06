from itertools import combinations
from typing import Dict, List, Union

from networkx.classes.graph import Graph

from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]) -> None:
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if isinstance(net, str):
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        elif isinstance(net, BayesNet):
            self.bn = net
        else:
            raise TypeError("net must be of type `str` or `BayesNet`")

    # TODO: This is where your methods should go
    def order(self, heuristic="mindeg", ascending=True) -> List[str]:
        adjacency = self.adjacency(self.bn.get_interaction_graph())
        order = []

        # Check whether given heuristic is valid
        heuristics = {
            h.split("_")[-1] for h in dir(self) if h.startswith("_order_heuristic_")
        }
        assert heuristic in heuristics, f"heuristic must be one of {heuristics}"
        # Determine the function that depicts our selection heuristic
        order_func = getattr(self, f"_order_heuristic_{heuristic}")
        # Select minimum if we want ascending order, otherwise maximum
        select = min if ascending else max

        for _ in range(len(adjacency)):
            # Select the node to eliminate from G based on heuristic
            v = select(adjacency, key=lambda x: order_func(adjacency, x))
            # Remove v from G
            self._elim_adjacency(adjacency, v)
            # Add v to list of ordering
            order.append(v)

        return order

    @staticmethod
    def _order_heuristic_mindeg(adjacency, node) -> int:
        # Mindeg is just the amount of neighbors a node has
        return len(adjacency[node])

    @staticmethod
    def _order_heuristic_minfill(adjacency: Dict[str, set], node: str) -> int:
        # Minfill is the amount of edges necessary to remove a node
        e = 0
        for u, v in combinations(adjacency[node], 2):
            if u not in adjacency[v]:
                e += 1

        return e

    @staticmethod
    def adjacency(G: Graph) -> Dict[str, set]:
        # Get a dict with nodes as key and its adjacent nodes as set
        return {v: set(G[v]) - set([v]) for v in G}

    @staticmethod
    def _elim_adjacency(adjacency: Dict[str, set], node: str):
        # Eliminate a variable and (re)add necessary edges
        neighbors = adjacency[node]
        new_edges = set()

        for u, v in combinations(neighbors, 2):
            if v not in adjacency[u]:
                adjacency[u].add(v)
                adjacency[v].add(u)
                new_edges.add((u, v))
                new_edges.add((v, u))

        for v in neighbors:
            adjacency[v].discard(node)

        del adjacency[node]