#from _typeshed import Self
from copy import deepcopy
from itertools import combinations, product
from typing import Dict, List, Optional, Union
from networkx.algorithms.shortest_paths.generic import has_path

import pandas as pd
from networkx.classes.graph import Graph
import networkx as nx
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

    def order(self, X=None, heuristic="mindeg", ascending=True) -> List[str]:
        G = self.bn.get_interaction_graph()
        all_nodes = self.bn.get_all_variables()

        if X is not None:
            for node in all_nodes:
                if node not in X:
                    G.remove_node(node)

        adjacency = self.adjacency(G)
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


    def joint_probability(self) -> pd.DataFrame:
        """Get the truth table with probabilities by chain rule"""
        # TODO: Perhaps make vars queryable to improve performance
        variables = self.bn.get_all_variables()

        # DataFrame with combinations of True and False per var
        truth_table = pd.DataFrame(
            product([True, False], repeat=len(variables)), columns=variables
        )

        for i, col in enumerate(truth_table):
            j = self.bn.get_cpt(col)
            # Determine what columns to merge on
            j_cols = j.columns
            j_cols = list(j_cols[j_cols != "p"])
            # Merge truth table with probabilities
            truth_table = truth_table.merge(j, on=j_cols)
            # Rename column with probability
            truth_table.rename({"p": f"p_{i}"}, inplace=True, axis=1)

        # Determine what columns have probabilities
        p_cols = [col for col in truth_table.columns if col.startswith("p_")]

        # Multiply all probabilities
        truth_table["p"] = 1
        for col in p_cols:
            truth_table["p"] = truth_table["p"] * truth_table[col]

        truth_table.drop(p_cols, axis=1, inplace=True)

        return truth_table

    def marginal_distribution(
        self,
        Q: Optional[List[str]] = None,
        E: Optional[Dict[str, bool]] = None,
    ) -> pd.DataFrame:
        probabilities = self.joint_probability()

        # If we only want specific vars then sum over all others
        if Q is not None:
            probabilities = (
                probabilities.groupby(Q).agg({"p": "sum"}).reset_index()
            )
        else:
            Q = self.bn.get_all_variables()

        if E is not None:
            # Make sure we can query given evidence
            for v in E:
                assert v in Q, f"evidence '{v}' not in {Q}"

            # Using pandas query and a query string
            # e.g. '`Winter?` == True and `Wet Grass?` == False'
            queries = [f"`{v}` == {e}" for v, e in E.items()]
            return probabilities.query(" and ".join(queries))

        return probabilities

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

    def map_mpe_estimation(
        self, 
        E: pd.Series,
        Q: Optional[List[str]] = None,
        heuristic: Optional[str] = "mindeg"
    ) -> pd.DataFrame:

        ## get all interesting variables:
        bn = self
        vars = bn.bn.get_all_variables()
        E_vars = []
        for i in range(0, len(E.index)):
                E_vars.append(E.index[i])

        # MAP?
        MAP = True

        # in case of MPE, Q = all variables not in E (for pruning)
        if not (Q):
            MAP = False
            Q = []
            for var in vars:
                if var not in E_vars:
                    Q.append(var)

        # 1. get order of elimination
        # pass argument of heuristic to order function
        # Assume: in case of MAP, the order of the MAP variables 
        #         is not changing after summing out the others?
        order = bn.order(heuristic = heuristic)

        # 2. prune the network as far as possible
        bn.pruning(Q, E)

        # 3. in case of MAP:
        # Multiply the factors and sum-out the variables not in Q and E, that exist after pruning

        if MAP:
            vars = bn.bn.get_all_variables()
            SumOut_Vars = []
            for var in vars:
                if var not in E_vars and var not in Q:
                    SumOut_Vars.append(var)

            for var in order:
                if var in SumOut_Vars:
                # get the factor (multiply them)
                # sum var out
                # update the cpt
                    order.remove(var)
                    ...
        
        else:
            bn.pruning_edges(E)
        

        # 4. maximise out
        #   MPE: all variables not in the evidenz set (MPE_Q)
        #   MAP: all variales in Q
        return "Hello world"