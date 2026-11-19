# Stress-TFM

Data-efficiency benchmark for tabular foundation models on physiological stress detection.

## Setup

    # Install uv (if not already installed)
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create venv and install the project
    uv venv --python 3.12
    source .venv/bin/activate
    uv sync

## Models

    Model       Pre-training           min_context
    ----------  ---------------------  -----------
    Kumo        Synthetic (SCM)        1
    Causilo     Synthetic              1
    TabDPT      Real (OpenML)          1
    TabICLv2    Synthetic              1
    TabPFN      Synthetic              1
    LightGBM    Trained from scratch   2

## Data

- Dataset: StressID (Chaptoukaev et al., NeurIPS 2023 D&B)
- 65 subjects, 11 tasks, 3 signals (ECG, EDA, Respiration) at 500 Hz
- 132 features per sample (46 ECG + 62 Respiration + 24 EDA)
- One row per (subject, task) pair (~710 rows total)
- Preprocessing: z-score + neurokit2 biosppy cleaning (official pipeline)


## Development

    # Lint
    uv run ruff check .

    # Format
    uv run ruff format .

    # Run pre-commit hooks
    uv run pre-commit run --all-files

    # Run tests
    uv run pytest tests/ -v

## Citations

- StressID: Chaptoukaev et al., "StressID: A Dataset for Stress Detection from Physiological Signals", NeurIPS 2023 Datasets and Benchmarks.
- TabPFN: Hollmann et al., "Accurate predictions on small data with a tabular foundation model", Nature, 2025.
- TabICL: Qu et al., "TabICL: In-context learning on tabular data", ICML 2025.
- Kumo: NVIDIA, "Kumo Tabular", 2026.
- Causilo: Nums AI, "Causilo", 2026.
- TabDPT: "TabDPT: A Denoising Probabilistic Transformer for tabular data", NeurIPS 2025.

## License

**(TBD)** Research use only.
