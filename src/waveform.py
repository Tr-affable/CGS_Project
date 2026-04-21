"""
Active Memory Burden waveform computation.

Implements the M(t) sliding-window algorithm from the CGS 410 proposal:

    At each word position t, compute the Active Memory Burden M(t) --
    the number of dependency edges that are "open" (one endpoint has
    been encountered but the other has not yet been reached) when
    processing words in left-to-right (linear) order.

The result is a 1-D time-series waveform:

    W = [M(t_1), M(t_2), ..., M(t_n)]

Uses an O(V + E) difference-array / prefix-sum technique so that
every edge is visited exactly once, and the final waveform is built
in a single linear pass.
"""

from typing import List
import networkx as nx


def compute_memory_burden(tree: nx.DiGraph, root: int) -> List[int]:
    """Return the Active Memory Burden waveform for *tree*.

    Parameters
    ----------
    tree : nx.DiGraph
        A dependency tree whose edges run head -> dependent.
        Node IDs must be numeric and their natural ordering defines
        the linear (left-to-right) word order in the sentence.
    root : int
        The ID of the (abstract) root node.  This node is excluded
        from the waveform -- only "real" word positions contribute.

    Returns
    -------
    list[int]
        W = [M(t_1), M(t_2), ..., M(t_n)] where each M(t_i) is the
        count of dependency edges that are open at word position t_i.

    Algorithm
    ---------
    A dependency edge (h, d) with h != root spans the linear interval
    [lo, hi) where lo = min(h, d) and hi = max(h, d).  The edge is
    "open" -- i.e. occupying working memory -- at every word position
    t that falls in [lo, hi):

        * At position lo the edge *enters* working memory (the first
          of its two endpoints is encountered).
        * At position hi the edge *leaves* working memory (the second
          endpoint finally appears, closing the dependency).

    Instead of counting open edges per position with a naive O(V*E)
    loop, we use a difference-array trick:

        diff[lo] += 1
        diff[hi] -= 1

    A single prefix-sum pass over *diff* then yields M(t) at every
    position in O(V) time, for a total complexity of O(V + E).

    Example
    -------
    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0, 2), (2, 1), (2, 4), (4, 3)])
    >>> compute_memory_burden(G, root=0)
    [1, 1, 2, 0]
    """
    # ----- 1. collect non-root nodes in linear (left-to-right) order -----
    nodes = sorted(n for n in tree.nodes() if n != root)
    if not nodes:
        return []

    n = len(nodes)

    # fast look-up: node ID  ->  0-based index in the sorted array
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # ----- 2. build the difference array in O(E) -----
    #  One extra slot at index n handles the right-boundary decrement
    #  without a special-case check.
    diff = [0] * (n + 1)

    for head, dep in tree.edges():
        # skip edges from the abstract root -- they are structural
        # artefacts, not real dependencies held in working memory
        if head == root:
            continue

        # both endpoints must be real words
        if head not in node_to_idx or dep not in node_to_idx:
            continue

        lo = min(node_to_idx[head], node_to_idx[dep])
        hi = max(node_to_idx[head], node_to_idx[dep])

        # edge is open at positions lo, lo+1, ..., hi-1
        diff[lo] += 1
        diff[hi] -= 1

    # ----- 3. prefix-sum in O(V) to obtain the waveform -----
    waveform = []
    running = 0
    for i in range(n):
        running += diff[i]
        waveform.append(running)

    return waveform
