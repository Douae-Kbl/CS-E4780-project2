# Enhancing LLM Inference with GraphRAG: Nobel Laureates

This project implements a robust **Graph Retrieval-Augmented Generation (GraphRAG)** system using **KuzuDB** and **DSPy**. It is designed to answer complex natural language questions about Nobel Laureates by converting them into accurate Cypher queries, executing them against a local knowledge graph, and generating grounded responses.

The system was developed for the **CS-E4780 Scalable Systems and Data Management** course at Aalto University.

## 🚀 Key Features

### Task 1: Robust Text2Cypher Pipeline
* **Dynamic Schema Pruning:** Reduces token usage and hallucinations by filtering the graph schema to only relevant node/edge types before query generation.
* **Few-Shot Retrieval (RAG-on-RAG):** Uses **ChromaDB** to retrieve the top-3 most similar valid Cypher examples based on the user's question to guide the LLM.
* **Self-Refinement Loop:** An iterative repair mechanism that uses Kuzu's `EXPLAIN` output to catch syntax errors and auto-correct queries before execution.
* **Post-Processing:** Rule-based enforcement of lowercase comparisons and correct property projections.

### Task 2: Performance Optimization
* **LRU Caching:** Implements a Least Recently Used cache to store question/answer pairs, reducing latency for repeated queries to ~0.02s.
* **End-to-End Latency Tracking:** Detailed breakdown of execution time per component (Pruning, Generation, Execution, etc.).

## 🛠️ Architecture

The pipeline leverages a separation of concerns:
1.  **Graph Construction:** Ingests JSON data into KuzuDB.
2.  **Retrieval:** Embeds user query -> Fetches few-shot examples (ChromaDB).
3.  **Generation:** DSPy Chain-of-Thought generates Cypher.
4.  **Validation:** `EXPLAIN` check -> Refinement Loop if error.
5.  **Execution:** Runs query on Kuzu.
6.  **Response:** LLM generates natural language answer from graph results.

## 📦 Tech Stack

* **Database:** [Kuzu](https://kuzudb.com/) (Embedded Graph DB)
* **Orchestration:** [DSPy](https://dspy.ai/)
* **LLM Provider:** Gemini (via OpenRouter)
* **Embeddings:** OpenAI `text-embedding-large`
* **Vector Store:** [ChromaDB](https://www.trychroma.com/)
* **Frontend/Runtime:** Marimo Notebook

## ⚙️ Setup & Installation

We recommend using the `uv` package manager
to manage dependencies.

```bash
# Uses the local pyproject.toml to add dependencies
uv sync
# Or, add them manually
uv add marimo dspy kuzu polars pyarrow
# Don't forget to source virtual env
source .venv/Scripts/activate
```

### Start Kuzu Database
```bash
docker compose up
```
Go to `localhost:8000` you can check the UI of the database

### Create basic graph
marimo simultaneously serves three functions. You can run Python code as a script, a notebook, or as an app!

### Open and Run the Initial GraphRAG system on a single query
```bash
# Open a marimo notebook in edit mode
marimo edit initial_system.py
# Run and open a marimo notebook in edit mode
uv run marimo edit initial_system.py
```

### Open and Run the Optimized GraphRAG system on a single query
```bash
# Open a marimo notebook in edit mode
marimo edit graph_rag.py
# Run and open a marimo notebook in edit mode
uv run marimo edit graph_rag.py
```

### Open and Run the Optimized GraphRAG system on the whole dataset
```bash
# Open a marimo notebook in edit mode
marimo edit testing_pipeline.py
# Run and open a marimo notebook in edit mode
uv run marimo edit testing_pipeline.py
```

## 💡 Other relevant files
[dataset](./data/generate_examples/nobel_questions_queries.csv) - Dataset .csv file.

[result_analysis.ipynb](result_analysis.ipynb) - A notebook that analyses the results of running the pipeline on our dataset : latency evaluation, caching metrics, accuracy computations, list of queries - generated queries and their correctness.

[testing pipeline notebook output](./data/test_results_with_caching) - A csv file containing the predicted Cypher queries and the generated answers.


## 📊 Evaluation Results

| Metric | Baseline | **Ours** |
| :--- | :--- | :--- |
| **Accuracy** (N=40) | 30% (12/40) | **90% (36/40)** |

## 👥 Contributors

* **Douae Kabelma** (Aalto University)
* **Cristiana Cocheci** (Aalto University)

