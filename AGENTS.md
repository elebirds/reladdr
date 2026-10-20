<!-- 说明 RelAddr 的开发边界、模块职责与必须遵循的检查。 -->
# Working on RelAddr

## Current stage

The shared runtime is implemented with configuration validation, local run records,
artifacts, provenance, CLI tools, and optional explicit MLflow sync/retry.
A three-shape engineering smoke in B01_Behavioral_Baseline has deterministic scenes,
ground truth, a Qwen2-VL Instruct BF16 adapter and local result recording. Its formal
runner adds stable scene-family and reference-position/relation splits, five layouts,
paired visual controls and clustered statistics. B02_Binding_Replication,
M01_Interface_Diagnostics and M02_Path_Diagnostics have native no-cache activation
patching, explicit token roles, layer sweeps, reversible hooks and suffix mediation. K-cache set search is not
implemented. Preserve the scientific meaning of the smoke and formal baseline while using only the canonical interfaces.
Stage/experiment IDs are authoritative in docs/research/06_experimental_design.md;
Experiment names use B/M/V/S identifiers plus English names (for example
B03_Reference_Disambiguation); the registry maps historical E IDs without rewriting runs.
Implemented experiments live in lowercase matching subpackages. Use only canonical IDs,
protocol versions, configs and module entry points; do not add legacy compatibility layers.
Historical run artifacts and their provenance retain the executed IDs and scientific meaning.
Do not create empty packages for planned work.
Detailed protocols live at docs/research/protocols/<full_experiment_name>.md.
The next-batch checklist and budget are authoritative in docs/research/08_resources_and_reproducibility.md.
Keep runtime behavior separate from experiment-specific scientific protocols. Add real
functionality with tests; do not create dummy experiment entry points.

## Environment

- Use the configured PATH and `uv`; the project pins Python in `.python-version`.
- Use `uv run --locked` for project commands and `uv sync --locked` to restore the
  environment. Update `pyproject.toml` and `uv.lock` together for dependency changes.
- `tracking` is an optional MLflow client extra; `oss` adds boto3 only for direct
  object-store uploads. Parameters/metrics and artifact policy are independent; new
  configs explicitly use `artifacts: "off"`. Proxy mode must match the remote run
  artifact URI; never silently fall back or reconfigure the server.
- `gpu` pins Torch 2.8.0, torchvision 0.23.0, Transformers 4.57.1 and Accelerate 1.11.0.
  Do not install this extra on the Mac just to run CPU tests. Image rendering uses Pillow.
- Keep project dependencies local. Do not modify shared shell/runtime configuration.

## Chinese comments

Handwritten source, tests, and configuration files start with a Chinese description.
Use Chinese comments for important contracts and non-obvious logic, especially state
transitions, data integrity, and synchronization. Keep generated lockfiles machine-managed.

## Module boundaries

- `reladdr.runtime`: config, lifecycle, local records, and optional tracking.
- `reladdr.experiments`: per-experiment subpackages for private schemas, comparisons, execution and analyses; `_shared` contains only genuinely shared execution.
- `reladdr.tasks`: scene semantics, rendering, questions, labels, and grouped splits.
- `reladdr.models`: model loading, processor/token maps, native tensor sites, and hooks.
- `reladdr.interventions`: cache capture, replacement, and reliable state restoration.
- `reladdr.mechanisms`: address representations, operators, and native-path analysis.
- `reladdr.evaluation`: behavioral/intermediate metrics and measured compute costs.
- `reladdr.search`: intervention-set objectives, baselines, and fractional optimization.

Keep model-specific GQA, position encoding, and cache-layout knowledge in one adapter.
Do not copy hook implementations into experiment scripts. Mechanism discovery and
set-search performance are separate claims even when they share infrastructure.

## Research records and data

- `docs/research/` is the repository's research dossier. Update the relevant section
  and its direct dependents instead of maintaining duplicate narratives.
- Distinguish author-reported evidence, mathematical deductions, working hypotheses,
  planned experiments, and locally measured results.
- Record model/data revisions, scene-grouped splits, seed, effective configuration,
  code state, dtype, attention backend, intervention site, outputs, and cost per run.
- Fit representations and select layers/hyperparameters without using the held-out
  evaluation labels. Never split questions from one scene across discovery and test.
- Probe decodability and head ablation are not evidence of a particular native operator.
- Preserve local run results before optional remote tracking. Report synchronization
  failures explicitly; do not silently discard results or claim a remote upload succeeded.

## Security and execution

Do not commit credentials, real `.env` files, generated datasets, checkpoints, large
activations, or run outputs. Do not log credentials or the full process environment.
Read secrets only through the approved environment/secret mechanism. Paid machines,
model downloads, remote service changes, publication, and Git pushes require relevant
user authorization; setting up this scaffold does not authorize them.

## Checks

```sh
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv run --locked python -c "import reladdr"
uv build
```

Tests use temporary directories and never download models or contact remote services.
The optional SDK test uses an explicitly enabled temporary file store only.
Run `uv run --locked --extra tracking pytest tests/runtime/test_tracking_sdk.py` when
changing the SDK integration; remote authentication and compatibility need live validation.
Run the actual CLI and build checks as well as tests when changing its entry points.
Optional tracking validation must not contact a server without relevant authorization.
