# Lang2LTL-SemPathBench

This directory contains the Lang2LTL adapter for SemPathBench. The default Gemini and debug paths do not depend on the original `Lang2LTL/` project. When the T5 symbolic translator from the paper is selected, the adapter lazily imports the original decoder from `Lang2LTL/`.

[Original Paper](https://openreview.net/pdf?id=VxfjGZzrdn) | [Official Repository](https://github.com/h2r/Lang2LTL)

## 1. Running the Method

### Environment Setup

```bash
conda create -n lang2ltl python=3.9 dill matplotlib plotly scipy scikit-learn pandas tenacity
conda activate lang2ltl
pip install tiktoken
pip install nltk seaborn pyyaml
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # GPU
pip install tensorboard transformers datasets evaluate torchtext
conda install -c conda-forge spot
```

### Model and API-Key Configuration

The LLM backend is Gemini. Configure it in the local key file:

```python
# scripts/methods/api_key.py
MODEL = "gemini-3.5-flash"
API_KEY = "your_gemini_api_key_here"
```

The symbolic translator uses the T5 checkpoint from the original paper. The runner checks `--symbolic-model-path` and downloads the checkpoint with `gdown` when it is missing. An existing directory containing `config.json`, model weights, and tokenizer files is reused without downloading it again.

| Component | Configuration | Automatic download |
| --- | --- | --- |
| RER / LLM prompt | Gemini, read from `MODEL` / `API_KEY` by default | No; API call |
| REG embedding | Gemini `gemini-embedding-001` by default | No; API call |
| Symbolic translator | Original T5-base `checkpoint-best` | Yes; Google Drive checkpoint |
| Planner | AP-MDP-style / A* planner in the SemPathBench adapter | Not applicable |

### Minimal Run

Run one instruction first:

```bash
python scripts/methods/lang2ltl/run.py --limit 1 --overwrite
```

Full `valunseen` example:

```bash
python scripts/methods/lang2ltl/run.py \
  --set valunseen \
  --translation-mode llm \
  --grounding-mode embedding \
  --planning-mode vanilla \
  --overwrite --overwrite-llm-cache --verbose
```

### Arguments

| Argument | Description |
| --- | --- |
| `--input-root PATH` | Instruction input directory. Defaults to the SemPathBench instruction root. |
| `--output-root PATH` | Directory for predictions and `summary.json`. Defaults to `resources/methods/baselines/Lang2LTL`. |
| `--set {valunseen,train,all}` | Instruction split to run. Defaults to `valunseen`. |
| `--llm-cache-root PATH` | LLM and translation cache. Defaults to `resources/methods/baselines/Lang2LTL/_llm_cache`. |
| `--embedding-cache-root PATH` | Cache for AP-description and referring-expression embeddings. Defaults to `resources/methods/baselines/Lang2LTL/_embedding_cache`. |
| `--model MODEL` | Gemini model. Defaults to `MODEL` in `scripts/methods/api_key.py`, then `GEMINI_MODEL`. |
| `--symbolic-model-path PATH` | T5 `checkpoint-best` directory. Defaults to `~/ground/models/lang2ltl/t5-base/checkpoint-best`. |
| `--symbolic-checkpoint-url URL` | T5 checkpoint URL. Defaults to the composed-dataset checkpoint folder linked by the original Lang2LTL README and is downloaded when missing. |
| `--translation-mode {auto,llm,heuristic}` | Translation mode. `llm` runs RER, grounding, and symbolic-LTL translation; `heuristic` uses local category matching; `auto` tries the LLM and falls back to the heuristic. |
| `--grounding-mode {embedding,local}` | REG mode. `embedding` ranks referring expressions and AP descriptions by cosine similarity; `local` is the legacy structured-matching ablation/debug mode. Defaults to `embedding`. |
| `--embedding-model MODEL` | Gemini embedding model. Defaults to `gemini-embedding-001`. |
| `--overwrite` | Replace existing predictions; otherwise, existing results are skipped. |
| `--overwrite-llm-cache` | Ignore cached translations and translate again. Recommended after changing the model or prompt. |
| `--max-prompt-objects N` | Maximum number of object APs in the prompt. Defaults to `120`. |
| `--object-label-radius N` | Radius over which an object AP is active on the grid. Defaults to `20`. |
| `--topk-groundings N` | Top-k grounding candidates retained for each referring expression. Defaults to `3`. |
| `--planning-mode {vanilla,fast,product}` | `vanilla` uses the AP-MDP-style planner; `product` uses the legacy product-graph A*; `fast` extracts eventual APs and connects them with standard A*. |
| `--limit N` | Run only the first N instructions for debugging. |
| `--verbose` | Print detailed map-inventory, planner, and search logs. |
| `--evaluate` | Deprecated compatibility option; metrics are always computed by the shared evaluator. |

## 2. Key Files

| File | Purpose |
| --- | --- |
| `run.py` | SemPathBench method runner, structured similarly to `ltlcodegen/run.py`. |
| `translator.py` | Lang2LTL-style RER, grounding, placeholder substitution, symbolic LTL, and caching; supports Gemini/OpenAI-compatible APIs and the original T5 symbolic translator. |
| `model_checkpoint.py` | Checks and automatically downloads the original T5 checkpoint. |
| `map_features.py` | Builds the room/object instance AP inventory from SemPathBench `map_state`. |
| `ltl.py` | Lightweight local LTLf formulas and progression semantics. |
| `ltl_parser.py` | Parses Lang2LTL/Spot-style prefix LTL such as `F a` and `& F a G ! b`. |
| `planner.py` | Builds the AP label map and runs AP-MDP-style planning, product-graph A*, or fast A*. |
| `__main__.py` | Supports `python -m scripts.methods.lang2ltl`. |
| `__init__.py` | Marks the directory as a Python package. |

## 3. Adaptation to SemPathBench

This directory contains the method-level changes required to adapt the original Lang2LTL approach to SemPathBench.

### 3.1 Input Format

The original Lang2LTL method uses its own benchmark format. This runner reads the SemPathBench instruction, `map_state`, `start_pose`, and task metadata.

### 3.2 Instance-Level APs

The original grounding is closer to landmark/name-level grounding. This adapter uses concrete SemPathBench instance APs, `room_i` and `object_i`, such as `room_3` and `object_23`.

### 3.3 AP Inventory

`map_features.py` extracts rooms, objects, categories, labels, attributes, and object-room relations from `map_state` to build the candidate AP inventory.

### 3.4 Object Descriptions

Object embedding descriptions are generated from map metadata rather than by RER. For example, a `basketball` in a `bedroom` is serialized as `a basketball located in a bedroom`.

### 3.5 Embedding-Based REG

The default grounding preserves Lang2LTL-style REG. The referring expression extracted by RER and every AP description are encoded with the same embedding model, and cosine similarity selects the top-ranked AP.

### 3.6 Symbolic LTL

The Lang2LTL sequence is preserved: RER -> REG -> grounded utterance -> symbolic LTL -> grounded LTL. Symbolic propositions avoid characters such as `i` that conflict with LTL operators.

### 3.7 LTL Parsing and Repair

The adapter adds a prefix-LTL parser. If an LLM response is incomplete, it attempts repair. If repair still fails, it creates a simple sequential-eventual formula from the grounded APs so that batch evaluation can continue.

### 3.8 Planner

The original Lang2LTL primarily produces grounded LTL, while SemPathBench requires a trajectory. The adapter maps `room_i` and `object_i` to grid cells and searches for a satisfying path. The default `vanilla` mode first uses LTL progression at the AP abstraction level to find an AP visitation plan and then realizes every AP action with grid A*. The `product` mode retains the legacy `(row, col, residual_formula)` product-state A*. The `fast` mode is a sequential-A* debug/ablation path.

### 3.9 Output

The output includes the prediction JSON, trajectory, shared-evaluator metrics, and grounding, translation, and planning metadata required by SemPathBench.
