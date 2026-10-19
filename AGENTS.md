# Working on RelAddr

## Current stage

This repository is a research scaffold. Python modules are empty package markers.
No task generator, model adapter, intervention, search algorithm, experiment CLI,
or MLflow logging integration has been implemented. Add real functionality and
its tests together when that implementation is requested; do not add dummy APIs
or always-passing tests to make the scaffold look complete.

## Environment

- Use the configured PATH and `uv`; the project pins Python in `.python-version`.
- Use `uv run --locked` for project commands and `uv sync --locked` to restore the
  environment. Update `pyproject.toml` and `uv.lock` together for dependency changes.
- `tracking` is an optional MLflow client extra. GPU dependencies are not installed
  by the scaffold; select their CUDA build and versions when implementing the adapter.
- Keep project dependencies local. Do not modify shared shell/runtime configuration.

## Module boundaries

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
uv run --locked python -c "import reladdr"
uv build
```

When implementation tests exist, also run `uv run --locked pytest`. Currently no
tests are collected; do not report the absence of tests as a passing research check.
Optional tracking validation imports the client only and must not contact a server.
