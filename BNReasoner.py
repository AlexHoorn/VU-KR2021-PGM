import random
from copy import deepcopy
from itertools import combinations, product
from typing import Dict, List, Optional, Set, Union

import networkx as nx
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

    # TODO: This is where your methods should go
    def d_sep_wrong(self, Xset, Ev, Yset,) -> bool: 
        for x in Xset:
            for y in Yset:
                print(x)
                paths = list(nx.all_shortest_paths(self.bn.get_interaction_graph(), x,y))
                for path in paths:
                    print(path)
                    if BNReasoner.block(path, Ev) == False:
                        return False
        return True
    
    def block_wrong(self, p, Ev,) -> bool:
        for step in range(0,len(p)-2):
            if (p[step+1] in self.bn.get_children(p[step]) and p[step+2] in self.bn.get_children(p[step+1])):
                if p[step+1] not in Ev:
                    return False
            #converging valve
            if p[step+1] in self.bn.get_children(p[step]) and p[step+1] in self.bn.get_children(p[step+2]):
                if p[step+1] in Ev or nx.descendants(p[step+1]) in Ev:
                    return False
            #diverging valve
            if p[step] in self.bn.get_children(p[step+1]) and p[step+2] in self.bn.get_children(p[step+1]):
                return False
        return True 

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
        adjacency = self.adjacency(G)

        # Remove the nodes we're not interested in
        not_X = [node for node in nodes if node not in X]
        for node in not_X:
            self._elim_adjacency(adjacency, node)

        # Check whether given heuristic is valid
        heuristics = ["random","mindeg","minfill"]
        assert heuristic in heuristics, f"heuristic must be one of {heuristics}"

        # Determine the function that depicts our selection heuristic
        order_func = getattr(self, f"_order_heuristic_{heuristic}")
        
        # Select minimum if we want ascending order, otherwise maximum
        select = min if ascending else max

        order = []
        for _ in range(len(adjacency)):
            # Select the node to eliminate from G based on heuristic
            v = select(adjacency, key=lambda x: order_func(adjacency, x))
            # Remove v from G
            self._elim_adjacency(adjacency, v)
            # Add v to list of ordering
            order.append(v)

        return order

    @staticmethod
    def _order_heuristic_random(*args) -> float:
        return random.uniform()

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
    def adjacency(G: Graph) -> Dict[str, set]:
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
        ## prune a network for a given query and evidence set as far as possible

        # combined set of states
        L = deepcopy(Q)
        for i in range(0, len(E.index)):
            if E.index[i] not in L:
                L.append(E.index[i])

        # first prune the leaves
        # repeat this as often as possible
        simpl = True
        while(simpl):
            V = self.bn.get_all_variables()
            count = 0
            for i in range(len(V)):
                if V[i] not in L:
                    if len(self.bn.get_children(V[i])) == 0:
                        self.bn.del_var(V[i])
                        count += 1
            if count == 0:
                simpl = False
                
        # adjust the CPTs
        L = []
        for i in range(0, len(E.index)):
            L.append(E.index[i])

        for i, node in enumerate(L):
            childs = self.bn.get_children(node)
            for child in childs:
                # only parent instantiation
                # newcpt = self.bn.get_compatible_instantiations_table(E.take([i]), self.bn.get_cpt(child))
                # all instantiations
                newcpt = self.bn.get_compatible_instantiations_table(E, self.bn.get_cpt(child))
                self.bn.update_cpt(child, newcpt)
            # simplify also all CPTs of the evidenz itself --> STATE THAT IN THE REPORT MAYBE?
            newcpt = self.bn.get_compatible_instantiations_table(E, self.bn.get_cpt(node))
            self.bn.update_cpt(node, newcpt)

        # then prune the edges
        for node in L:
            childs = self.bn.get_children(node)
            for child in childs:
                self.bn.del_edge([node, child])

    def joint_probability(
        self, Q: Optional[List[str]] = None, E: Optional[Dict[str, bool]] = None
    ) -> pd.DataFrame:
        """Get the truth table with probabilities by chain rule"""
        nodes = self.bn.get_all_variables()

        if E is None:
            E = {}

        # DataFrame with combinations of True and False per var
        cpt = pd.DataFrame(product([True, False], repeat=len(nodes)), columns=nodes)

        for node in nodes:
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

        return cpt

    def _merge_predecessors(self, var: str, E: Dict[str, bool]):
        cpt = self.get_cpt_evidence(var, E)
        cpt = cpt.rename({"p": f"p_{var}"}, axis=1)
        # This assumes that a cpt always has itself and p as the last 2 columns
        preds = cpt.columns[:-2]

        # Return cpt if no predecessors
        if len(preds) == 0:
            return cpt

        # Recursively get cpt for every predecessors by merging with their own predecessors
        pred_cpts = (self._merge_predecessors(p, E) for p in preds)

        # Merge the cpts of the predecessors
        for pred_cpt in pred_cpts:
            cpt = cpt.merge(pred_cpt)

        return cpt

    # This gets ALL predecessors in the whole path, not just the direct predecessors of the node
    def get_predecessors(self, q: str) -> Set[str]:
        G = self.bn.structure
        tree_vars = set([q])

        while True:
            cache_tree_vars = tree_vars.copy()
            for var in cache_tree_vars:
                for var in G.predecessors(var):
                    tree_vars.add(var)

            if tree_vars == cache_tree_vars:
                break

        return tree_vars

    def get_cpt_evidence(self, variable: str, E: pd.Series):
        return self._query_cpt(self.bn.get_cpt(variable), E)

    @staticmethod
    def _query_cpt(cpt: DataFrame, query: pd.Series):
        if query is None:
            return cpt

        for q, v in query.items():
            if q in cpt.columns:
                cpt = cpt[cpt[q] == v]

        return cpt

    def d_separation_with_pruning(self, X, Z, Y):
        # copy the graph 
        P = deepcopy(self)

        # delete leaf nodes
        deletion = True

        while deletion == True:
            count = 0
            V = P.bn.get_all_variables()
            for i in range(len(V)):
                if V[i] not in X and V[i] not in Y and V[i] not in Z:
                    if len(P.bn.get_children(V[i])) == 0:
                            P.bn.del_var(V[i])
                            count += 1
           # print('after deleting leaf nodes: '+ str(P.bn.get_all_variables()))
            if count == 0:
                deletion = False

        # delete outgoing edges from Z
        for var in Z:
            childs = P.bn.get_children(var)
            for child in childs:
                P.bn.del_edge([var, child])

        #check for every node in X and Y if there is a connection (connection = )
        # if so, X and Y are not d-separated by Z

        for x in X:
            for y in Y:
               if nx.has_path(nx.to_undirected(P.bn.structure), x,y):
                    return False
        return True
    
    ## TO DO: MAP and MPE estimation
