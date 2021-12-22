import random
from copy import deepcopy
from functools import reduce
from itertools import combinations, product
from operator import mul
from typing import Dict, List, Optional, Set, Union

import networkx as nx
import numpy as np
import pandas as pd
from networkx.classes.graph import Graph
from pandas.core.frame import DataFrame

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

    def order(
        self,
        X: Optional[List[str]] = None,
        heuristic: str = "mindeg",
        ascending: bool = True,
    ) -> List[str]:
        # Get undirected Graph
        G = self.bn.structure.to_undirected()
        nodes = self.bn.get_all_variables()

        # Get all vars if X is None
        if X is None:
            X = nodes.copy()

        # Adjacency dict where every key is a node and the items its neighbors
        adjacency = self._adjacency(G)

        # Remove the nodes we're not interested in
        not_X = [node for node in nodes if node not in X]
        for node in not_X:
            self._elim_adjacency(adjacency, node)

        # Check whether given heuristic is valid
        heuristics = ["random", "mindeg", "minfill"]
        assert heuristic in heuristics, f"heuristic must be one of {heuristics}"

        # Determine the function that depicts our selection heuristic
        order_func = getattr(self, f"_order_heuristic_{heuristic}")

        # Select minimum if we want ascending order, otherwise maximum
        select = min if ascending else max

        order = []
        # Adj changes during iteration so iterate over range of len
        for _ in range(len(adjacency)):
            # Select the node to eliminate from G based on heuristic
            v = select(adjacency, key=lambda x: order_func(adjacency, x))
            # Remove v from G
            self._elim_adjacency(adjacency, v)
            # Add v to list of ordering
            order.append(v)

        return order

    @staticmethod
    def _order_heuristic_random(*_) -> float:
        return random.uniform(0, 1)

    @staticmethod
    def _order_heuristic_mindeg(adjacency, node) -> int:
        # Mindeg is the amount of neighbors a node has
        return len(adjacency[node])

    @staticmethod
    def _order_heuristic_minfill(adjacency: Dict[str, set], node: str) -> int:
        # Minfill is the amount of edges that need to be added to remove a node
        e = 0
        for u, v in combinations(adjacency[node], 2):
            if u not in adjacency[v]:
                e += 1

        return e

    @staticmethod
    def _adjacency(G: Graph) -> Dict[str, set]:
        # Get a dict with nodes as key and its adjacent nodes as set
        return {v: set(G[v]) - set([v]) for v in G}

    @staticmethod
    def _elim_adjacency(adjacency: Dict[str, set], node: str) -> None:
        # Eliminate a variable and add edges between its neighbors
        # this is done inplace of the given adjacency dict, so no new dict is returned
        neighbors = adjacency[node]

        # Create edges between all neighbors
        for u, v in combinations(neighbors, 2):
            if v not in adjacency[u]:
                adjacency[u].add(v)
                adjacency[v].add(u)

        # Remove mention of node in every neighbor
        for v in neighbors:
            adjacency[v].discard(node)

        # Remove node from adjacency dict
        del adjacency[node]

    def pruning(self, Q: List[str], E: pd.Series) -> None:
        # Prune a network for a given query and evidence set as far as possible

        # Combined set of states
        L = set(Q) | set(E.index)

        # First prune the leaves repeat this as often as possible
        while True:
            V = set(self.bn.get_all_variables())
            count = 0

            # For every v not in L
            for v in V - L:
                if len(self.bn.get_children(v)) == 0:
                    self.bn.del_var(v)
                    count += 1

            if count == 0:
                break

        # Adjust the CPTs
        L = set(E.index)

        for node in set(E.index):
            childs = self.bn.get_children(node)

            for child in childs:
                # All instantiations
                newcpt = self.bn.get_compatible_instantiations_table(
                    E, self.bn.get_cpt(child)
                )
                self.bn.update_cpt(child, newcpt)

            # Simplify also all CPTs of the evidenz itself
            newcpt = self.bn.get_compatible_instantiations_table(
                E, self.bn.get_cpt(node)
            )
            self.bn.update_cpt(node, newcpt)

        # Then prune the edges
        for node in L:
            childs = self.bn.get_children(node)

            for child in childs:
                self.bn.del_edge((node, child))

    def joint_probability(
        self, Q: Optional[List[str]] = None, E: Optional[Dict[str, bool]] = None
    ) -> pd.DataFrame:
        """Get the truth table with probabilities by chain rule"""
        nodes = self.bn.get_all_variables()

        if E is None:
            E = {}

        # DataFrame with combinations of True and False per var
        cpt = pd.DataFrame(product([True, False], repeat=len(nodes)), columns=nodes)
        cpt = self._query_cpt(cpt, E)

        for node in list(cpt.columns):
            # Merge truth table with probabilities
            cpt = cpt.merge(self.get_cpt_evidence(node, E))
            # Rename column with probability
            cpt = cpt.rename({"p": f"p_{node}"}, axis=1)

        # Eliminate rows with 0
        cpt = cpt[~(cpt.select_dtypes("number") == 0).any(axis=1)]

        # Multiply all probabilities
        p_cols = cpt.select_dtypes("number").columns
        cpt["p"] = cpt[p_cols].product(axis=1)
        cpt.drop(p_cols, axis=1, inplace=True)

        # Remove columns with evidence
        cpt = cpt.drop(list(E.keys()), axis=1)

        # Sum by Q
        if Q is not None:
            cpt = cpt.groupby(Q)["p"].sum().reset_index()

        # Normalize by sum
        cpt["p"] = cpt["p"] / cpt["p"].sum()

        # Apply tidy sorting
        cpt = cpt.sort_values(list(cpt.columns), ascending=False)
        cpt = cpt.reset_index(drop=True)

        return cpt

    def marginal_distribution(
        self, Q: List[str], E: Optional[Dict[str, bool]] = None,
    ) -> pd.DataFrame:
        # Create empty evidence if not given
        if E is None:
            E = {}

        # Construct end result for every Q union E
        Q_E = set(Q) | set(E)
        cpt = pd.DataFrame(product([True, False], repeat=len(Q_E)), columns=Q_E)

        for q in Q_E:
            # Iteratively merge every q
            q_cpt = self._merge_predecessors(q, E)
            cpt = cpt.merge(q_cpt)

        # Eliminate rows with 0
        cpt = cpt[~(cpt.select_dtypes("number") == 0).any(axis=1)]

        # Calculate probability
        cpt["p"] = cpt.select_dtypes("number").product(axis=1)

        # Sum probabilities by Q
        cpt = cpt.groupby(Q).agg({"p": "sum"}).reset_index()

        # Normalize probability by sum
        cpt["p"] = cpt["p"] / cpt["p"].sum()

        # Apply tidy sorting
        cpt = cpt.sort_values(list(cpt.columns), ascending=False)
        cpt = cpt.reset_index(drop=True)

        return cpt

    def _merge_predecessors(self, var: str, E: Dict[str, bool]):
        # This assumes that a cpt always has itself and p as the last 2 columns
        preds = list(self.bn.structure.predecessors(var))

        # Return cpt if no predecessors
        if len(preds) == 0:
            return self.get_cpt_evidence(var, E, rename_p=True)

        # Recursively get cpt for every predecessors by merging with their own predecessors
        pred_cpts = (self._merge_predecessors(p, E) for p in preds)

        # Merge the cpt with cpts of the predecessors
        cpt = self.get_cpt_evidence(var, E, rename_p=True)
        for pred_cpt in pred_cpts:
            cpt = cpt.merge(pred_cpt)

        return cpt

    def get_cpt_evidence(self, variable: str, E: Dict[str, bool], rename_p=False):
        cpt = self._query_cpt(self.bn.get_cpt(variable), E)

        if rename_p:
            cpt = cpt.rename({"p": f"p_{variable}"}, axis=1)

        return cpt

    @staticmethod
    def _query_cpt(cpt: DataFrame, query: Dict[str, bool]):
        if query in [None, {}]:
            return cpt

        for q in set(cpt.columns) & set(query):
            cpt = cpt[cpt[q] == query[q]]

        return cpt

    def d_separation(self, X: List[str], Z: List[str], Y: List[str]):
        # Copy the graph
        P = deepcopy(self)

        # Delete leaf nodes
        while True:
            V = set(P.bn.get_all_variables())
            count = 0

            for v in V - set(X + Y + Z):
                if len(P.bn.get_children(v)) == 0:
                    P.bn.del_var(v)
                    count += 1

            if count == 0:
                break

        # Delete outgoing edges from Z
        for var in Z:
            childs = P.bn.get_children(var)

            for child in childs:
                P.bn.del_edge((var, child))

        # Check for every node in X and Y if there is a connection if so, X and Y are not d-separated by Z
        for x, y in product(X, Y):
            if nx.has_path(nx.to_undirected(P.bn.structure), x, y):
                return False

        return True

    def map_mpe_estimation(
        self, E: pd.Series, Q: Optional[List[str]] = [], heuristic: str = "mindeg"
    ) -> pd.DataFrame:

        # Get all interesting variables:
        bn = deepcopy(self)
        vars = set(bn.bn.get_all_variables())
        E_vars = set(E.index)

        # MAP?
        MAP = bool(Q)

        Q = set(Q)

        # In case of MPE, Q = all variables not in E (for pruning)
        if not MAP:
            Q = vars - E_vars

        # 1. Prune the network as far as possible
        bn.pruning(list(Q), E)

        # 2. Get order of elimination
        order = bn.order(heuristic=heuristic)

        # 3. In case of MAP:
        # Multiply the factors and sum-out the variables not in Q and E, that exist after pruning
        CPT = bn.bn.get_all_cpts()

        if MAP:
            vars = set(bn.bn.get_all_variables())
            SumOut_Vars = vars - Q

            # We often use sets to decrease necessary looping
            # but here ordering is important so we can't
            for node in order:
                if node in SumOut_Vars:
                    # Get the new factor
                    used_cpts, newcpt, cols = self.multiplication_factors(CPT, node, E)

                    # Sum var out if any
                    sum_cols = list(cols - {node})
                    if sum_cols:
                        newcpt = newcpt.groupby(sum_cols)["p"].sum().reset_index()

                    # Update + replace other factors
                    CPT[node] = newcpt  # not referring to the name, thus this works
                    for var in used_cpts:
                        if var != node:
                            del CPT[var]

            # Update order
            order = list(set(order) - SumOut_Vars)

        # 4. maximise out
        for node in order:
            # Get new factor, all used cpts and the col names
            used_cpts, newcpt, cols = self.multiplication_factors(CPT, node, E)
            # Maximise out
            newcpt = self.maximise_out(newcpt, cols, node)

            # Maybe evidenz is already clear?
            if len(np.unique(newcpt[node].values)) == 1:
                if node not in E.index:
                    E = E.append(pd.Series(newcpt.iloc[0][node], {node}))

            # Update + replace other factors
            CPT[node] = newcpt  # not referring to the name, thus this works
            for var in used_cpts:
                if var != node:
                    del CPT[var]

        # Build the result
        result = self.build_result(CPT)

        return result

    @staticmethod
    def build_result(CPT: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Result: maybe there is more than one instantiation
        vmax = reduce(mul, [cpt.shape[0] for cpt in CPT.values()], 1)

        # Make all single independent cpts containing as many rows as max
        for var, cpt in CPT.items():
            newcpt = cpt

            while newcpt.shape[0] < vmax:
                newcpt = newcpt.append(cpt, ignore_index=True)

            CPT[var] = newcpt

        data = {}
        p = []

        for cpt in CPT.values():
            for col in cpt.columns:
                if col == "p":
                    p.append(list(cpt[col].values))

                else:
                    data[col] = cpt[col]

        p = np.asmatrix(p)
        p = np.prod(p, axis=0)
        p = p.tolist()
        p = [item for sublist in p for item in sublist]

        result = pd.DataFrame(data)
        result["p"] = p

        result = result.query("p == p.max()").reset_index(drop=True)

        return result

    @staticmethod
    def multiplication_factors(CPT: Dict[str, pd.DataFrame], node: str, E: pd.Series):

        # Select all cpts that are used and set the col names
        used_cpts = {}
        cols = {node}

        for cpt in CPT:
            cpt_cols = (set(CPT[cpt].columns)) - set("p")
            if node in cpt_cols:
                used_cpts[cpt] = CPT[cpt]
                cols = cols.union(cpt_cols)

        # Create newcpt for the faktor. Shorten it already as far as possible/needed
        newcpt = pd.DataFrame(product([False, True], repeat=len(cols)), columns=cols)
        newcpt["p"] = 1.0
        newcpt = BayesNet.get_compatible_instantiations_table(E, newcpt)
        newcpt = newcpt.reset_index(drop=True)
        newcpt["valid"] = True

        # Multiply all faktors + check if rows are still needed
        for i in range(newcpt.shape[0]):
            row = pd.Series(newcpt.iloc[i][:-2], cols)  # as instantiation

            for cpt in used_cpts.values():
                p = BayesNet.get_compatible_instantiations_table(row, cpt)["p"]

                if len(p) == 0:
                    newcpt.at[i, "valid"] = False
                    break

                p = p.values[0]
                newcpt.at[i, "p"] *= p

        newcpt = newcpt.loc[newcpt["valid"]].reset_index(drop=True)
        del newcpt["valid"]

        return used_cpts, newcpt, cols

    @staticmethod
    def maximise_out(newcpt: pd.DataFrame, cols: Set[str], node: str) -> pd.DataFrame:

        # Max out and get all maxima
        if len(cols) > 1:
            maxcpt = newcpt.groupby(list(cols - {node}))["p"].max().reset_index()
            fillcpt = []

            for i in range(maxcpt.shape[0]):
                inst = pd.Series(maxcpt.iloc[i], maxcpt.columns)  # as instantiation
                row = BayesNet.get_compatible_instantiations_table(inst, newcpt)
                fillcpt.append(row)

            newcpt = pd.concat(fillcpt, ignore_index=True)

        else:
            newcpt = newcpt.query("p == p.max()").reset_index(drop=True)

        return newcpt
