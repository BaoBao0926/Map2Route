# Lang2LTL-2 SemPathBench

This directory connects the original `Lang2LTL-2/` implementation to the SemPathBench method runner.

The adapter is text-only by default. SemPathBench does not provide an image input equivalent to the original Spot/OSM landmark-image database, so the visual-landmark branch is disabled.

## 1. Running the Method

### Environment Setup

The original Lang2LTL-2 implementation depends on PyTorch, Transformers, Spot/GraphNav packages, and several scientific-computing libraries. The SemPathBench adapter replaces the OpenAI chat and embedding calls in `openai_models.py` with a Gemini-compatible layer. At minimum, install:

```bash
conda create -n lang2ltl python=3.9 dill matplotlib plotly scipy scikit-learn pandas tenacity
conda activate lang2ltl
pip install tiktoken
pip install nltk seaborn pyyaml
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # GPU
pip install tensorboard transformers datasets evaluate torchtext
conda install -c conda-forge spot
pip install dill numpy scipy scikit-learn matplotlib tqdm pyproj utm
pip install torch transformers
```

The Boston Dynamics SDK is also required to use the original Spot graph loader in `spg.py/load_map.py`. The SemPathBench bridge preferentially generates `obj_locs.json`, so most text-only debugging does not require real Spot graph files.

### Gemini Model and API Key

SRER and REG use Gemini for LLM and embedding calls by default. Configure the local key file:

```python
# scripts/methods/api_key.py
MODEL = "gemini-3.5-flash"
API_KEY = "your_gemini_api_key_here"
```

### LT Model

The original Lang2LTL-2 LT module requires a fine-tuned T5 checkpoint by default:

```text
~/ground/models/checkpoint-best
```

Specify another location with:

```bash
--lt-model-path /path/to/checkpoint-best
```

If the checkpoint is unavailable locally, the runner can download it from the Google Drive link in the original Lang2LTL-2 README:

```bash
python scripts/methods/lang2ltlv2/run.py \
  --download-lt-checkpoint \
  --overwrite-cache \
  --limit 1 \
  --verbose
```

Automatic download uses `gdown`; if it is missing, the runner first attempts to install it. After the download, the `--lt-model-path` directory must contain Hugging Face checkpoint files such as `config.json`, `pytorch_model.bin` or `model.safetensors`, plus tokenizer files.

### Minimal Run

Start with the `fast` planner to verify the adapter. It first uses `grounded_ltl` produced by the original Lang2LTL-2 pipeline. If LT dependencies or the checkpoint are missing and no `grounded_ltl` is available, it falls back to the AP order grounded by SRER/REG/SPG and connects those APs with sequential A*.

```bash
python scripts/methods/lang2ltlv2/run.py \
  --planner fast \
  --limit 1 \
  --overwrite \
  --overwrite-cache \
  --verbose
```

The `vanilla` mode uses the AP-MDP-style planner implemented by this adapter. It searches for an acceptable AP visitation plan with LTL progression and realizes every AP action using grid A*. The implementation reuses the AP label map, LTL progression, and A* helpers in this repository. If AP-MDP produces no movement trajectory for complex or contradictory LTL, the runner falls back to sequential A* and records `planner_type=ap_mdp_with_fast_fallback` in metadata.

```bash
python scripts/methods/lang2ltlv2/run.py \
  --planner vanilla \
  --limit 1 \
  --overwrite \
  --verbose
```

Full rerun example that replaces earlier fallback predictions and translation caches:

```bash
conda activate lang2ltl
python scripts/methods/lang2ltlv2/run.py \
  --set valunseen \
  --workers 1 \
  --overwrite \
  --overwrite-cache
```

`--overwrite` replaces existing predictions. `--overwrite-cache` rebuilds the translation cache so that the original SRER/REG/SPG/LT (T5) pipeline runs again.

## 2. Arguments

| Argument | Description |
| --- | --- |
| `--input-root PATH` | Instruction input directory. Defaults to the SemPathBench instruction root. |
| `--output-root PATH` | Directory for predictions and `summary.json`. Defaults to `resources/methods/baselines/Lang2LTL2`. |
| `--set {valunseen,train,all}` | Instruction split to run. Defaults to `valunseen`. |
| `--cache-root PATH` | Cache for pseudo maps, OSM data, and module outputs. Defaults to `resources/methods/baselines/Lang2LTL2/_sembench_cache`. |
| `--lt-model-path PATH` | Fine-tuned T5 checkpoint from the original Lang2LTL-2 method. Defaults to `~/ground/models/checkpoint-best`. |
| `--download-lt-checkpoint` | Download the LT checkpoint from the original Google Drive link if `--lt-model-path` does not exist. |
| `--lt-checkpoint-url URL` | Google Drive folder URL for the LT checkpoint. Defaults to the link in the original Lang2LTL-2 README. |
| `--model MODEL` | Gemini chat model. Defaults to `MODEL` in `scripts/methods/api_key.py`, then `GEMINI_MODEL`. |
| `--embedding-model MODEL` | Gemini embedding model. Defaults to `EMBEDDING_MODEL`, then `GEMINI_EMBEDDING_MODEL`. |
| `--planner {vanilla,fast}` | `vanilla` uses AP-MDP-style planning with sequential-A* fallback; `fast` directly uses the sequential-A* debug planner. Defaults to `fast`. |
| `--topk-groundings N` | Number of top REG/SPG grounding candidates to retain. Defaults to `10`. |
| `--object-label-radius N` | Radius of object AP labels in the `fast` planner. Defaults to `20`. |
| `--overwrite` | Replace existing predictions; otherwise, existing results are skipped. |
| `--overwrite-cache` | Rebuild the bridge and module caches. |
| `--limit N` | Run only the first N instructions for debugging. |
| `--verbose` | Print detailed execution logs. |
| `--evaluate` | Deprecated compatibility option; metrics are always computed by the shared evaluator. |

## 3. Output

The default output directory is:

```text
resources/methods/baselines/Lang2LTL2
```

Each prediction contains:

```text
lang2ltlv2.translation
lang2ltlv2.bridge
lang2ltlv2.planner
lang2ltlv2.planner_details
```

These fields record SRER/REG/SPG/LT outputs, failures, pseudo-environment bridge information, and planner status.
