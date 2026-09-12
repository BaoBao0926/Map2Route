# Tutorial Method

This is a simple SemPathBench baseline. It chooses nearby traversable cells for
ordered hard must-pass constraints, connects them with A*, writes predictions,
evaluates them, and saves a trajectory PNG next to each JSON result.

Run one example:

```bash
python scripts/methods/tutorial/run.py --limit 1
```

Run the default set:

```bash
python scripts/methods/tutorial/run.py
```

Run all instructions:

```bash
python scripts/methods/tutorial/run.py --set all
```

Default output:

```text
resources/methods/baselines/tutorial
```

Each prediction writes:

```text
resources/methods/baselines/tutorial/<map_id>/<instruction_id>.json
resources/methods/baselines/tutorial/<map_id>/<instruction_id>.png
resources/methods/baselines/tutorial/summary.json
```

Each JSON prediction records `runtime_seconds`, and `summary.json` contains a
`runtime_summary` with total, average, minimum, and maximum episode runtime.

## Parameters

| Parameter | Description |
| --- | --- |
| `--input-root PATH` | Instruction input directory. Defaults to `resources/instructions`. |
| `--output-root PATH` | Output directory for predictions and `summary.json`. Defaults to `resources/methods/baselines/tutorial`. |
| `--set {valunseen,train,all}` | Instruction set to run. Defaults to `valunseen`; use `all` to run everything. |
| `--resume` | Reuse existing prediction files. Without this flag, all selected episodes are regenerated. |
| `--evaluate` | Deprecated compatibility flag. Metrics are always computed. |
| `--limit N` | Run only the first `N` matching instructions. Useful for testing. |
