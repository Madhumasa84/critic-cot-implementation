# Critic-CoT Data Engineering Layer

This folder contains the systemized version of the Critic-CoT notebook.

## Modules

- `storage/reasoning_db.py`: SQLite storage for traces, steps, critiques, and daily metrics.
- `pipeline/data_ingestion.py`: GSM8K ingestion, caching, CSV helpers, and sample normalization.
- `pipeline/critic_cot_wrapper.py`: OpenRouter calls, answer extraction, arithmetic verification, and the four reasoning strategies.
- `pipeline/simple_pipeline.py`: Main runner that executes strategies, stores traces, and exports result files.
- `pipeline/run_evaluation.py`: CLI evaluation script with configurable sample size and automated report generation.
- `pipeline/scheduler.py`: One-time or recurring scheduled runs with logging to `logs/daily_results.csv`.

## Configuration

You can provide credentials in one of these ways:

1. Set `OPENROUTER_API_KEY` and optionally `MODEL` as environment variables.
2. Copy `config.example.py` to `config.py` in the project root and fill in your values.
3. Create a `.env` file in the project root with `OPENROUTER_API_KEY=...`.

## Example Commands

Run an evaluation:

```powershell
python .\data_engineering\pipeline\run_evaluation.py --samples 20 --strategies baseline,iter_refine,filter,majority
```

Run the scheduler once:

```powershell
python .\data_engineering\pipeline\scheduler.py --mode once --samples 20
```

Run the scheduler continuously every day at 09:00:

```powershell
python .\data_engineering\pipeline\scheduler.py --mode continuous --time 09:00 --samples 20
```
