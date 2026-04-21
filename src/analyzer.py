import numpy as np
import networkx as nx

def get_tree_features(tree: nx.DiGraph, root: int, ncross: int):
    """Extract all 8 graph parameters from a single tree."""
    out_degrees = [d for n, d in tree.out_degree() if n != root]
    max_arity = max(out_degrees) if out_degrees else 0
    avg_arity = np.mean(out_degrees) if out_degrees else 0
    
    lengths = nx.shortest_path_length(tree, root)
    max_depth = max(lengths.values()) if lengths else 0
    
    graph_density = nx.density(tree)
    
    ic_list, dl_list = [], []
    lr_edges = 0
    valid_edges = 0
    
    for u, v in tree.edges():
        if u == root: continue
        valid_edges += 1
        if u < v: lr_edges += 1
        lower, upper = min(u, v), max(u, v)
        ic = 1
        dl = 0
        for nodex in tree.nodes():
            if nodex == root: continue
            if lower < nodex < upper:
                dl += 1
                if tree.out_degree(nodex) > 0:
                    ic += 1
        ic_list.append(ic)
        dl_list.append(dl)
    
    avg_ic = np.mean(ic_list) if ic_list else 0
    avg_dl = np.mean(dl_list) if dl_list else 0
    directionality = lr_edges / valid_edges if valid_edges > 0 else 0
    
    return [max_arity, avg_arity, max_depth, graph_density, avg_ic, avg_dl, directionality, ncross]
