import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import pearsonr

def get_bpe_to_word_mapping(sentence, tokenizer):
    """
    Phase 1: Maps BPE tokens back to their original word indices.
    Uses the Fast Tokenizer's word_ids() for scientific precision.
    """
    # Ensure it's a fast tokenizer
    assert tokenizer.is_fast, "Must use a Fast tokenizer for accurate word mapping."
    
    encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    word_ids = encoded.word_ids(batch_index=0)
    
    token_to_word = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is not None:
            token_to_word[token_idx] = word_idx
            
    num_words = max(token_to_word.values()) + 1
    return encoded, token_to_word, num_words

def extract_attention_matrices(encoded_inputs, model):
    """
    Phase 2: Extracts the attention tensors from all layers.
    Returns: A list of numpy arrays, one for each layer.
             Each array is of shape (N, N), averaged across heads.
    """
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        
    # outputs.attentions is a tuple of (batch_size, num_heads, seq_len, seq_len) tensors
    all_layers_attention = []
    for layer_attention in outputs.attentions:
        # Squeeze batch dim, shape becomes (num_heads, N, N)
        layer_matrix = layer_attention.squeeze(0)
        # Average across heads
        avg_head_matrix = layer_matrix.mean(dim=0).float().cpu().numpy()
        all_layers_attention.append(avg_head_matrix)
        
    return all_layers_attention

def pool_attention_matrix(attention_matrix, token_to_word, num_words):
    """
    Phase 3: Compress N x N token matrix to V x V word matrix.
    Uses sum pooling as probability aggregations across sub-words.
    """
    v_matrix = np.zeros((num_words, num_words))
    num_tokens = attention_matrix.shape[0]
    
    for i in range(num_tokens):
        for j in range(num_tokens):
            word_i = token_to_word.get(i)
            word_j = token_to_word.get(j)
            
            # Transformers attend to themselves, but self-loops are invalid in dependency trees.
            # We enforce 0 for self-attention at the word level.
            if word_i is not None and word_j is not None and word_i != word_j:
                v_matrix[word_i, word_j] += attention_matrix[i, j]
                
    return v_matrix

def build_dependency_tree(v_matrix):
    """
    Phase 4: Chu-Liu/Edmonds Algorithm to find Maximum Spanning Arborescence.
    Converts continuous attention probabilities into a discrete, cycle-free DAG.
    """
    # Create directed graph
    G = nx.DiGraph()
    num_words = v_matrix.shape[0]
    
    # Add nodes and edges
    for i in range(num_words):
        G.add_node(i)
        for j in range(num_words):
            if i != j and v_matrix[i, j] > 0:
                # NetworkX MSA maximizes the 'weight' attribute.
                G.add_edge(i, j, weight=v_matrix[i, j])
                
    # Run the algorithm
    msa = nx.algorithms.tree.branchings.maximum_spanning_arborescence(G)
    return msa

def calculate_memory_burden(tree, num_words):
    """
    Phase 5a: $O(V+E)$ sliding window to calculate active memory burden M(t).
    An edge (u, v) is "active" at time t if min(u,v) <= t and max(u,v) > t.
    """
    M_t = np.zeros(num_words)
    
    # Iterate through time steps (each word read)
    for t in range(num_words):
        active_dependencies = 0
        for u, v in tree.edges():
            # A dependency is open if the start word has been seen, 
            # but the end word has not yet been processed (or vice versa for head-final).
            start, end = min(u, v), max(u, v)
            if start <= t < end:
                active_dependencies += 1
        M_t[t] = active_dependencies
        
    return M_t

def calculate_uas(llm_tree, human_tree):
    """
    Calculates Unlabeled Attachment Score (UAS) between the generated tree and the gold standard human tree.
    UAS is the percentage of words that have the correct head assigned.
    """
    llm_edges = set(llm_tree.edges())
    human_edges = set(human_tree.edges())
    
    if len(human_edges) == 0:
        return 0.0
        
    correct_edges = len(llm_edges.intersection(human_edges))
    return correct_edges / len(human_edges)

def interpolate_and_evaluate(llm_waveform, human_waveform, num_bins=20):
    """
    Phase 5b: Interpolates to 20 bins and calculates Pearson correlation.
    """
    # Interpolate LLM
    x_llm = np.linspace(0, 1, len(llm_waveform))
    x_bins = np.linspace(0, 1, num_bins)
    llm_interp = np.interp(x_bins, x_llm, llm_waveform)
    
    # Interpolate Human (if not already 20 bins)
    if len(human_waveform) != num_bins:
        x_human = np.linspace(0, 1, len(human_waveform))
        human_interp = np.interp(x_bins, x_human, human_waveform)
    else:
        human_interp = human_waveform
        
    # Calculate Pearson R and p-value
    # Using scipy.stats.pearsonr
    r_stat, p_value = pearsonr(llm_interp, human_interp)
    return llm_interp, human_interp, r_stat, p_value

def run_pipeline(sentence, model_name="gpt2", human_baseline=None):
    """
    Orchestrates the entire upgraded scientific pipeline.
    """
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    
    print("Phase 1: BPE Alignment...")
    encoded_inputs, token_map, num_words = get_bpe_to_word_mapping(sentence, tokenizer)
    print(f"-> Sentence mapped to {num_words} words.")
    
    print("Phase 2: Extracting Attentions (All Layers)...")
    all_layers = extract_attention_matrices(encoded_inputs, model)
    
    # For demonstration, we will analyze Layer 6 (middle layer, 0-indexed as 5)
    layer_idx = min(5, len(all_layers)-1)
    print(f"Phase 3: Pooling Matrix for Layer {layer_idx+1}...")
    v_matrix = pool_attention_matrix(all_layers[layer_idx], token_map, num_words)
    
    print("Phase 4: Running Chu-Liu/Edmonds Arborescence...")
    dependency_tree = build_dependency_tree(v_matrix)
    print(f"-> Generated Tree Edges: {dependency_tree.edges()}")
    
    print("Phase 5: Calculating Memory Burden M(t)...")
    M_t = calculate_memory_burden(dependency_tree, num_words)
    print(f"-> M(t) Waveform: {M_t}")
    
    if human_baseline is not None:
        llm_interp, human_interp, r, p = interpolate_and_evaluate(M_t, human_baseline)
        
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 100, 20), human_interp, label="Human SUD Baseline", linestyle='--', color='gray', linewidth=2)
        plt.plot(np.linspace(0, 100, 20), llm_interp, label=f"LLM ({model_name})", color='blue', linewidth=2)
        
        plt.title(f"Memory Burden: LLM vs Human\nPearson r: {r:.3f} (p-value: {p:.3e})", fontsize=14)
        plt.xlabel("Sentence Progress (%)", fontsize=12)
        plt.ylabel("Active Memory Dependencies M(t)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("pipeline_result.png", dpi=300)
        print("-> Plot saved to pipeline_result.png")
        
    return M_t, dependency_tree, v_matrix

if __name__ == "__main__":
    # Test Sentence
    test_sentence = "The scientist analyzed the complex data carefully."
    
    # Dummy Human Baseline for testing (e.g., matching a theoretical expectation)
    # This dummy array matches the length of the sentence (7 words).
    dummy_human = np.array([0, 1, 2, 3, 2, 1, 0])
    
    print("==================================================")
    print(f"TESTING PIPELINE WITH SENTENCE: '{test_sentence}'")
    print("==================================================")
    run_pipeline(test_sentence, model_name="gpt2", human_baseline=dummy_human)
