# Run the GitHub project on Kaggle

Use **Qwen3Merger_GitHub_Kaggle.ipynb** when you want Kaggle to download and run
the project from GitHub. Unlike the original self-contained notebook, this small
launcher fetches the source snapshot at runtime.

1. Create a Kaggle notebook and import/upload `Qwen3Merger_GitHub_Kaggle.ipynb`.
2. Enable Internet and select GPU T4 ×2.
3. Review the first code cell. The default runs preparation, benchmark preflight,
   training, ranking/export, all three pilot benchmarks, and reporting.
4. Run all cells. Logs stream into the notebook. The final cells display actual
   tables, charts, the Markdown report, and an analysis ZIP download.

The launcher pins source commit `f59a42245e20a126c80965445de88c5af3f21a52`.
It preserves Kaggle's installed CUDA PyTorch and installs project dependencies
in a separate environment. Each experiment stage uses a separate process to
release GPU memory on exit. No Hugging Face or GitHub token is required for these
public repositories under normal access conditions.

Outputs are written to `/kaggle/working/Qwen3Merger/`. The default training budget
is seven hours; ranking and benchmarking require additional time and may need
later Kaggle sessions. This is not a promise that the full pipeline fits one session.

## Resume in a later session

Save the entire output folder with Kaggle Save Version. Attach it as an input to
your new notebook, and set `RESTORE_FROM` to the directory containing
`manifests/run.json` and `checkpoints/`. Keep the source commit and training settings
the same. The launcher refuses to merge restored files into a populated output
directory; set `RESTORE_FROM=None` when that directory is already restored.

Useful stage selections:

```python
# Preparation and training only:
STAGES = ['prepare', 'benchmark-preflight', 'train']

# Ranking/export after restoring completed training:
STAGES = ['rank-export', 'report']

# Benchmarks after restoring model exports and the complete experiment folder:
STAGES = ['benchmark-preflight', 'benchmark', 'report']
BENCHMARKS = ['MMMLU', 'INCLUDE', 'MGSM']

# Display existing outputs:
STAGES = []
RUN_TESTS = False
```

`OVERRIDES['evaluation_mode'] = 'full'` selects full evaluation with the same
trained models. Pilot and full predictions are separate. `VARIANT` can select
one model when splitting benchmark work; finish matching baseline predictions
before expecting paired comparison tables.

The analysis ZIP excludes checkpoints and model weights to avoid doubling large
files. Preserve `checkpoints/` and `model_exports/` separately for resume/deployment.
Stage logs and execution history are saved under `launcher/`.

## Validation

All eight code cells compile. Four local launcher tests passed, covering snapshot
extraction, protection of modified source files, stage ordering, and failure
logging. Run `python -m pytest -q launcher_tests` to repeat these tests.
Generate the notebook with `python tools/build_github_launcher.py`.

The launcher has not been executed on Kaggle's GPUs here. Model training and
benchmark scores remain pending until the experiment actually runs there.
