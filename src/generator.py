import random
import networkx as nx
from typing import List, Tuple, Optional

# ── Prüfer code → undirected tree (from treegen.py) ─────────────────

def random_pruefer_code(n: int) -> List[int]:
    """Generate a random Prüfer code of length n-2 for a tree with n nodes."""
    return [random.choice(range(n)) for _ in range(n - 2)]


def tree_edges_from_pruefer_code(code: List[int]):
    """Decode a Prüfer code into edges of an undirected tree."""
    code = list(code)
    l1 = set(range(len(code) + 2))
    edges = []
    while code and len(l1) > 2:
        x = min(l1.difference(code))
        edges.append((x, code[0]))
        l1.remove(x)
        code.pop(0)
    assert len(l1) == 2
    edges.append(tuple(l1))  # final edge between the two remaining nodes
    return edges


def tree_from_pruefer_code(code: List[int]) -> nx.Graph:
    """Create an undirected tree from a Prüfer code."""
    return nx.Graph(tree_edges_from_pruefer_code(code))


# ── Root an undirected tree at a given node → directed tree ─────────

def rooted_at(undirected_tree: nx.Graph, root_node: int) -> nx.DiGraph:
    """
    Given an undirected tree and a root node, return a directed tree
    (DiGraph) where all edges point away from the root (head → dependent).
    Uses BFS from root_node.
    """
    dtree = nx.DiGraph()
    dtree.add_nodes_from(undirected_tree.nodes())
    visited = {root_node}
    queue = [root_node]
    while queue:
        node = queue.pop(0)
        for neighbor in undirected_tree.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                dtree.add_edge(node, neighbor)  # head → dependent
                queue.append(neighbor)
    return dtree


def all_directed_trees(undirected_tree: nx.Graph) -> List[nx.DiGraph]:
    """Given an undirected tree, return all possible rooted directed trees."""
    return [rooted_at(undirected_tree, node)
            for node in undirected_tree.nodes()]


# print("✓ Prüfer code utilities loaded.")

def is_projective_edge(tree: nx.DiGraph, edge: Tuple[int, int],
                       abstract_root: int) -> bool:
    """
    Check if a single edge (head, dep) in the tree is projective.

    An edge is projective iff every node linearly between head and dep
    is a descendant of the head (or its own head is inside the span).
    This mirrors Compute_measures.is_projective() from the reference code.
    """
    h, d = edge
    lo, hi = min(h, d), max(h, d)

    # Nodes linearly between head and dependent
    all_nodes = set(nx.descendants(tree, abstract_root))
    edge_span = [n for n in all_nodes if lo < n < hi]

    for nodeI in edge_span:
        node_head = tree.nodes[nodeI].get('head', None)
        if node_head is None:
            # For random trees without 'head' attr, find head via predecessors
            preds = list(tree.predecessors(nodeI))
            node_head = preds[0] if preds else abstract_root

        if node_head not in edge_span:
            if nodeI not in nx.descendants(tree, h):
                return False
    return True


def count_crossings(tree: nx.DiGraph, abstract_root: int) -> int:
    """Count the number of non-projective edges (excluding edges from abstract root)."""
    ncross = 0
    for edge in tree.edges:
        if edge[0] != abstract_root:
            if not is_projective_edge(tree, edge, abstract_root):
                ncross += 1
    return ncross


# print("✓ Crossing counter loaded.")

ABSTRACT_ROOT = 1000   # same as reference code


class RandomBaselineGenerator:
    """
    Generates a random baseline tree that matches a real tree in:
      - number of nodes (edges)
      - number of crossing (non-projective) edges

    Mirrors `Random_base` from baseline_conditions_random_structures.py.
    """

    def __init__(self, real_tree: nx.DiGraph, real_root: int = 0):
        self.real_tree = real_tree
        self.real_root = real_root
        self.n_edges = len(real_tree.edges)

        # Count crossings in the real tree
        self.num_cross_real = count_crossings(real_tree, real_root)

    def _try_one(self) -> Optional[nx.DiGraph]:
        """
        Generate one random Prüfer-code tree, try all rootings.
        Return the first directed tree whose crossing count matches,
        or None if no rooting works.
        """
        n = self.n_edges  # number of edges = number of real nodes (excl. abstract root)
        code = random_pruefer_code(n)
        undirected = tree_from_pruefer_code(code)
        candidates = all_directed_trees(undirected)
        random.shuffle(candidates)

        for dtree in candidates:
            # Find the root of this directed tree
            real_root_node = next(nx.topological_sort(dtree))

            # Add abstract root (1000) → real root edge
            dtree.add_edge(ABSTRACT_ROOT, real_root_node)

            # Set 'head' attribute on every node (needed by crossing checker)
            for h, d in dtree.edges:
                dtree.nodes[d]['head'] = h

            # Check if crossings match
            ncross_rand = count_crossings(dtree, ABSTRACT_ROOT)
            if ncross_rand == self.num_cross_real:
                return dtree

            # Remove the abstract root edge before trying next rooting
            dtree.remove_edge(ABSTRACT_ROOT, real_root_node)
            if ABSTRACT_ROOT in dtree.nodes:
                dtree.remove_node(ABSTRACT_ROOT)

        return None

    def generate(self, max_attempts: int = 40000) -> Optional[nx.DiGraph]:
        """
        Try up to max_attempts Prüfer codes to find a matching random tree.

        Returns
        -------
        nx.DiGraph or None
            A random directed tree with abstract root = 1000,
            matching the real tree in node count and crossing count.
        """
        for _ in range(max_attempts):
            result = self._try_one()
            if result is not None:
                return result
        return None  # could not find a match


# print("✓ RandomBaselineGenerator loaded.")

