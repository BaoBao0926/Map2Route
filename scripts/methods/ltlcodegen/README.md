# LTLCodeGen-SemPathBench

[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11246351) | [Official Repository](https://github.com/ExistentialRobotics/LTLCodeGen)

## 1. Running the Method

### Gemini Model and API Key

Configure the local key file before running the method:

```python
# scripts/methods/api_key.py
MODEL = "gemini-3.5-flash"
API_KEY = "your_gemini_api_key_here"
```

`scripts/methods/api_key.py` is listed in `.gitignore` and will not be committed.

### Minimal Run

Run one instruction first:

```bash
python scripts/methods/ltlcodegen/run.py --limit 1 --overwrite
```

Full run:

```bash
python scripts/methods/ltlcodegen/run.py \
  --set valunseen \
  --planning-mode vanilla --object-mode all \
  --overwrite
```

### Arguments

| Argument | Description |
| --- | --- |
| `--input-root PATH` | Instruction input directory. Defaults to the SemPathBench instruction root. |
| `--output-root PATH` | Directory for predictions and `summary.json`. Defaults to `resources/methods/baselines/LTLCodeGen`. |
| `--set {valunseen,train,all}` | Instruction split to run. Defaults to `valunseen`; use `all` for every split. |
| `--llm-cache-root PATH` | LLM translation cache. Defaults to `resources/methods/baselines/LTLCodeGen/_llm_cache`. |
| `--model MODEL` | Gemini model. Defaults to `MODEL` in `scripts/methods/api_key.py`, then `GEMINI_MODEL`, and finally `gemini-3.5-flash`. |
| `--overwrite` | Replace existing predictions; otherwise, existing results are skipped. |
| `--overwrite-llm-cache` | Ignore cached translations and call Gemini again. Recommended after changing the model. |
| `--max-prompt-objects N` | Maximum number of object APs included in the prompt. Defaults to `120`. |
| `--object-label-radius N` | Radius over which an object AP is active on the grid. Defaults to `20`. |
| `--planning-mode` / `--planning_mode {vanilla,fast}` | Planner mode. `vanilla` uses product-graph A*; `fast` extracts eventual APs in order and connects them with standard grid A*. Defaults to `vanilla`. The legacy `--mode` option remains available for compatibility. |
| `--object-mode` / `--object_mode {vanilla,all}` | Object-target mode. `vanilla` uses only the object AP selected by the LLM; `all` expands it to every instance in the same category and visits them in sequence. Defaults to `vanilla`; the misspelling `vinalla` is accepted for compatibility. |
| `--limit N` | Run only the first N instructions for debugging. |
| `--verbose` | Print detailed loading, AP-inventory, translation, planning, evaluation, and file-writing logs. |
| `--evaluate` | Deprecated compatibility option; metrics are always computed by the shared evaluator. |

### Output Path

```text
resources/methods/baselines/LTLCodeGen
```

## 2. Adaptation to SemPathBench

This directory contains the method-level changes required to adapt LTLCodeGen to SemPathBench.

### 2.1 LLM Call

Gemini performs LTLCodeGen translation. The default model is read from `MODEL` in `scripts/methods/api_key.py`, for example `gemini-3.5-flash`. Gemini generates Python code that defines `question()`, and the adapter executes `question()` to obtain the LTL formula.

### 2.2 Local LTL Library

The lightweight local `ltl.py` library supports APs, Boolean operators, `eventually`, `always`, `until`, and `next`. It represents automaton state through formula progression and therefore does not require the ROS/Spot environment used by the original LTLCodeGen implementation.

### 2.3 Planner

The planner converts a SemPathBench map into an AP label map. Room APs are true in the corresponding room cells, and object APs are true within a fixed radius of each object center. Search runs on the occupancy/traversability grid and returns a SemPathBench trajectory.

### 2.4 Planning Modes

The default `vanilla` mode follows the LTLCodeGen product-graph approach. Each search state is `(row, col, residual_formula)`, and every movement progresses the LTL formula until the residual formula is accepting.

The approximate `fast` mode extracts positive eventual APs in order and visits them using standard grid A*. It is usually faster but does not fully validate complex LTL semantics such as disjunction, `always`, `until`, or negative constraints.

### 2.5 Object Modes

The `vanilla` mode visits only the object AP selected by Gemini. For example, if Gemini produces `object_4`, only `object_4` becomes a target.

The `all` mode expands a selected AP by object category. If `object_4` is a chair and the map contains multiple chairs, the planner expands the target to every chair AP and visits them in sequence.

### 2.6 Inference Adaptation

The adapter builds its AP inventory only from information visible at inference time, including the instruction text, start pose, room layer, object-instance layer, and traversability. It does not use benchmark annotations such as `hard_constraints`, `soft_constraints`, ground-truth objects, or human trajectories for planning.

The planner already handles `start_pose`, so the prompt asks Gemini not to encode phrases such as "start from," "beside you," or "near you" as LTL APs. If the model still emits a top-level initial AP, the adapter removes it and records the change in `removed_initial_aps` metadata.
