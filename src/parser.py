import os
import networkx as nx
from typing import List, Dict, Tuple, Optional
from conllu import parse

def build_tree_from_sentence(sentence) -> nx.DiGraph:
    """
    Convert one parsed CoNLL-U sentence (from the `conllu` library) into a
    networkx DiGraph, following the DLM-ICM-baselines approach:

      1.  Each non-punctuation token becomes a node with attributes
          (form, lemma, upostag, xpostag, feats, head, deprel, deps, misc).
      2.  A virtual ROOT node (id = 0) is added.
      3.  Directed edges go from head → dependent.
      4.  Punctuation tokens (deprel == 'punct') are excluded.
    """
    tree = nx.DiGraph()

    for token in sentence:
        # Skip multi-word tokens (id is a tuple/range) and empty nodes
        if not isinstance(token['id'], int):
            continue;
        deprel = token['deprel']

        # ── Exclude punctuation (same as reference code) ──
        if deprel == 'punct':
            continue
        tree.add_node(
            token['id'],
            form    = token['form'],
            lemma   = token.get('lemma', '_'),
            upostag = token.get('upos', '_'),
            xpostag = token.get('xpos', '_'),
            feats   = token.get('feats', '_'),
            head    = token['head'] if token['head'] is not None else 0,
            deprel  = deprel,
            deps    = token.get('deps', '_'),
            misc    = token.get('misc', '_'),
        )

    # ── Add virtual ROOT (id = 0) ──
    ROOT = 0
    tree.add_node(ROOT)

    # ── Create directed edges (head → dependent) ──
    for nodex in list(tree.nodes):
        if nodex != ROOT:
            head_id = tree.nodes[nodex]['head']
            if tree.has_node(head_id):                    # handle disjoint trees
                tree.add_edge(
                    head_id,
                    nodex,
                    drel=tree.nodes[nodex]['deprel'],
                )

    return tree

MAX_EDGES = 12   # exclusive upper bound (n < 12)


def parse_conllu_file(filepath: str,
                      max_edges: int = MAX_EDGES
                      ) -> List[Dict]:
    """
    Read a .conllu file and return a list of dicts, each containing:
        - 'tree'    : nx.DiGraph  (the dependency tree)
        - 'sent_id' : int
        - 'text'    : str         (original sentence text if available)
        - 'n_edges' : int         (number of edges, excl. punct)

    Only sentences with  1 < n_edges < max_edges  are kept.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = f.read()

    sentences = parse(data)
    results: List[Dict] = []

    for idx, sentence in enumerate(sentences, start=1):
        tree = build_tree_from_sentence(sentence)
        n = len(tree.edges)

        if n < max_edges and n > 1:
            text = sentence.metadata.get('text', '') if sentence.metadata else ''
            results.append({
                'tree'   : tree,
                'sent_id': idx,
                'text'   : text,
                'n_edges': n,
            })

    return results


def parse_all_conllu_in_dir(directory: str,
                            max_edges: int = MAX_EDGES
                            ) -> Dict[str, List[Dict]]:
    """
    Walk a directory, find every .conllu file, parse and filter them.

    Returns
    -------
    dict :  { lang : [ {tree, sent_id, text, n_edges}, … ] }
    """
    corpus: Dict[str, List[Dict]] = {}
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            if fname.endswith('.conllu'):
                fpath = os.path.join(root, fname)
                trees = parse_conllu_file(fpath, max_edges)

                # ── Extract language name from filename ──
                lang = fname.replace('_train.conllu', '')
                lang = lang.replace('_test.conllu', '')
                lang = lang.replace('.conllu', '')

                corpus[lang] = trees
#                 print(f"  ✓ {lang:<20s}  ({fname})  →  {len(trees)} sentences  (edges < {max_edges})")
    return corpus

