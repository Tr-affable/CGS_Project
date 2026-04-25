# Decoding the Temporal and Topological Fingerprints of Human Language

**CGS 410 — Course Project**

This repository contains the code, data processing pipelines, and analytical notebooks for our research on the cognitive load of human language processing. We evaluate the Dependency Length Minimization (DLM) hypothesis by modeling sentence processing as a 1-D time series of Active Memory Burden, denoted as $M(t)$.

## Project Overview

When a person reads a sentence, their working memory is burdened by unfinished syntactic dependencies. Instead of collapsing this cognitive load into a single static scalar (like average dependency length), this project computes the **Active Memory Burden $M(t)$** dynamically at each word position. 

We pursue three main objectives in a unified pipeline:
1. **Real vs. Random Comparison**: Compare memory-burden waveforms of real human sentences against mathematically rigorous constraint-matched random trees across 10 diverse languages.
2. **Typology Classification**: Use machine learning (XGBoost) trained on topological and temporal features to classify sentences as Real vs. Random, and to classify the typological word order (SOV vs. SVO) in a strict Leave-One-Language-Out (LOLO) cross-validation setting.
3. **LLM Probe (Track 3)**: Investigate whether the internal attention mechanisms of Large Language Models (like GPT-2 and Qwen2.5-0.5B) reproduce the same cognitive-load waveform experienced by a human reader.

## Repository Structure

The project is structured into a highly modular pipeline:

### Core Source Code (`src/`)
* `parser.py`: Ingests Surface-syntactic Universal Dependencies (SUD) `.conllu` datasets into `networkx` graphs and applies rigorous node and punctuation filters.
* `generator.py`: Generates constraint-matched random baseline trees using uniform Prüfer-code sampling (Algorithm 1).
* `waveform.py`: The computational core calculating the Active Memory Burden $M(t)$ in $\mathcal{O}(V+E)$ time using a difference-array approach.
* `analyzer.py`: Computes 8 structural/topological features (e.g., Arity, Density, DLM, ICM, Edge Directionality).
* `data_processor.py`: Calculates 11 temporal features from the waveform (e.g., Peak Position, Change Rate, Skewness) and compiles the master feature matrix.
* `ml_engine.py`: Manages XGBoost model training, LOLO cross-validation folds, scoring, and SHAP explainability.
* `track3_pipeline.py`: Extracts LLM attention matrices, applies the Chu-Liu/Edmonds maximum spanning arborescence algorithm, and statistically correlates LLM implicit load with human baselines.

### Notebooks & Execution
* `CGS410_Project.ipynb`: The master interactive dashboard unifying the data loading, caching, machine learning evaluations, and the generation of primary analytical plots (Figures 1-4).
* `Track3_Analysis_Notebook.ipynb`: A specialized notebook explicitly dedicated to evaluating LLM attention dynamics and computing Unlabeled Attachment Scores (UAS).

### Data & Results
* `Random_Data/`: Directory storing binary cache (`.pkl`) files of computationally expensive random baseline generations to dramatically speed up reproducible execution.
* `Results_folder/`: The output destination containing all high-resolution evaluation figures ready for manuscript integration.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Tr-affable/CGS_Project.git
   cd CGS_Project
   ```

2. Unzip the dataset:
   Extract the `train.zip` archive into a `train/` directory in the root of the project.

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This will install all packages for the ML pipeline as well as `torch` and `transformers` required for the **Track 3 LLM Attention Probe**.)*

## Usage

1. **Main Pipeline**: Open `CGS410_Project.ipynb` and run the cells sequentially to execute the core data processing, feature extraction, and XGBoost classification models. The notebook leverages cached data from `Random_Data/` to run the entire ML pipeline in roughly 4 minutes.
2. **LLM Attention Probe**: Open `Track3_Analysis_Notebook.ipynb` to evaluate the GPT-2 (English) and Qwen (Hindi) causal language models against the human dependency tree baselines.

## Key Findings

* **Hypothesis 1 (Cognitive Load)**: Supported. Real human languages actively suppress memory burden, remaining strictly below the random baseline waveforms across all 10 evaluated languages.
* **Hypothesis 2 (Typology Shapes)**: Supported. SOV (head-final) and SVO (head-initial) languages exhibit distinctly different cognitive peak timings and amplitudes due to deferred vs. continuous dependency resolution.
* **Hypothesis 3 (LLM Alignment)**: Partially Supported. Causal LLM attention partially mimics the directional shape of human working memory but significantly underestimates the cumulative peak magnitude and peaks much earlier than humans do.

## Collaborators

* Tushar Bagani (241108)
* Rishitesh Kesri (240871)
* Divya Prakash Pandey (240369)
* Ineni Sree Charan (240464)
* Bhavesh Bisen (240267)
* Harsh Mehra (240431)
