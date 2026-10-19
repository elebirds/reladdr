<!-- 说明 RelAddr 的开发边界、模块职责与必须遵循的检查。 -->
# Working on RelAddr

## Current stage

The shared runtime is implemented with configuration validation, local run records,
artifacts, provenance, CLI tools, and optional explicit MLflow sync/retry.
Research packages remain descriptive markers: no scene generator, model adapter, interventions, or search algorithms exist.
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
- GPU dependencies are not installed
  by the scaffold; select their CUDA build and versions when implementing the adapter.
- Keep project dependencies local. Do not modify shared shell/runtime configuration.

## Chinese comments

Handwritten source, tests, and configuration files start with a Chinese description.
Use Chinese comments for important contracts and non-obvious logic, especially state
transitions, data integrity, and synchronization. Keep generated lockfiles machine-managed.

## Module boundaries

- `reladdr.runtime`: config, lifecycle, local records, and optional tracking.
- `reladdr.experiments` (when added): private schemas, comparisons, execution and analyses.
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
