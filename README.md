# stg_light_eval

Lightweight evaluation suite for spatio-temporal scene grammars targeting CVPR-style proofs with tiny datasets. The project focuses on rapid prototyping, sanity checking, and benchmarking of generative scene grammar models under resource-constrained experimental settings.

## Highlights
- Modular placeholders for synthetic data generation, evaluation metrics, and baseline adapters.
- Ready-to-extend utilities and notebook templates for rapid experiment turn-around.
- Packaging metadata, linting configs, and dependency lists to streamline collaboration.

## Repository Layout
```
stg_light_eval/
├── synthetic/              # Hooks for synthetic scene generators and toy grammars
├── data/                   # Dataset wrappers, loaders, and pre-processing scripts
├── metrics/                # Metric definitions and aggregation helpers
├── eval/                   # Experiment entry points and evaluation orchestrators
├── tables/                 # Result tables, LaTeX exports, and logging glue
├── notebooks/              # Colab-ready prototypes, reports, and sanity checks
├── utils/                  # Shared helpers (IO, geometry, viz, scheduling)
└── baselines/
    ├── gsplat/             # Gaussian splatting baselines
    ├── slotattn/           # Slot Attention alignment baselines
    └── pointtracks/        # Point tracking and correspondence baselines
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quickstart (Google Colab)
1. Open a new Colab notebook and enable GPU if available (`Runtime > Change runtime type > GPU`).
2. Clone this repository:
   ```python
   !git clone https://github.com/your-org/stg_light_eval.git
   %cd stg_light_eval
   ```
3. Install Python dependencies:
   ```python
   !pip install -r requirements.txt
   ```
4. (Optional) Install the package in editable mode to simplify imports:
   ```python
   !pip install -e .
   ```
5. Copy or author a notebook inside `notebooks/` and use the provided module scaffold (e.g., `from synthetic import toy_scene`) to prototype experiments.
6. Sync results back to your workspace by exporting metrics to `tables/` or pushing commits as needed.

## Contributing
- Run `flake8` and `isort` before submitting pull requests.
- Keep synthetic datasets and checkpoints lightweight; prefer procedural generation where possible.
- Document new evaluation protocols with minimal reproducible examples.

## License
Released under the MIT License. See `LICENSE` for details.
