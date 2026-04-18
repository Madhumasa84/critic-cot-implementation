"""
Command-line entrypoint for larger Critic-CoT evaluations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_engineering.config import settings
from data_engineering.pipeline.simple_pipeline import CriticCoTPipeline


def ensure_api_key_configured() -> None:
    if settings.has_configured_api_key():
        return
    raise RuntimeError(
        "OPENROUTER_API_KEY is not configured. Add it to config.py, .env, or your environment variables before running evaluation."
    )


def _parse_strategy_list(raw: str) -> List[str]:
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    if not requested:
        return list(settings.STRATEGIES)

    unknown = [strategy for strategy in requested if strategy not in settings.STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")
    return requested


def _build_strategy_configs(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    configs: Dict[str, Dict[str, int]] = {}
    for strategy in _parse_strategy_list(args.strategies):
        if strategy == "iter_refine":
            configs[strategy] = {"max_iterations": args.max_iterations}
        elif strategy == "filter":
            configs[strategy] = {"num_samples": args.filter_samples}
        elif strategy == "majority":
            configs[strategy] = {"num_samples": args.majority_samples}
        else:
            configs[strategy] = {}
    return configs


def _write_report_csv(path: Path, rows: List[Dict[str, object]]) -> str:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    return str(path)


def _write_markdown_report(
    path: Path,
    summary: Dict[str, Dict[str, object]],
    config: Dict[str, object],
) -> str:
    lines = [
        "# Critic-CoT Evaluation Report",
        "",
        f"- Generated: {datetime.now().isoformat()}",
        f"- Samples: {config['samples']}",
        f"- Split: {config['split']}",
        f"- Strategies: {', '.join(config['strategies'])}",
        "",
        "| Strategy | Accuracy (%) | Correct | Total | Avg Latency (ms) | Avg Tokens | Total Cost (USD) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for strategy in config["strategies"]:
        item = summary[strategy]
        lines.append(
            f"| {strategy} | {item['accuracy_pct']} | {item['correct_count']} | "
            f"{item['total_samples']} | {item['avg_latency_ms']} | {item['avg_tokens']} | "
            f"{item['total_cost_usd']} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def run_scaled_evaluation(
    num_samples: int = settings.DEFAULT_EVAL_SAMPLES,
    split: str = settings.DEFAULT_DATASET_SPLIT,
    shuffle: bool = False,
    seed: int = settings.DEFAULT_SAMPLE_SEED,
    strategies: str = ",".join(settings.STRATEGIES),
    max_iterations: int = settings.DEFAULT_MAX_ITERATIONS,
    filter_samples: int = settings.DEFAULT_FILTER_SAMPLES,
    majority_samples: int = settings.DEFAULT_MAJORITY_SAMPLES,
) -> Dict[str, object]:
    ensure_api_key_configured()
    pipeline = CriticCoTPipeline()
    samples = pipeline.ingestor.load_gsm8k(
        split=split,
        limit=num_samples,
        shuffle=shuffle,
        seed=seed,
    )

    strategy_configs = _build_strategy_configs(
        argparse.Namespace(
            strategies=strategies,
            max_iterations=max_iterations,
            filter_samples=filter_samples,
            majority_samples=majority_samples,
        )
    )

    summary = pipeline.run_all_strategies(samples, strategy_configs=strategy_configs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_rows = [summary[strategy] for strategy in strategy_configs]
    report_csv_path = settings.REPORT_DIR / f"evaluation_report_{timestamp}.csv"
    report_json_path = settings.REPORT_DIR / f"evaluation_report_{timestamp}.json"
    report_md_path = settings.REPORT_DIR / f"evaluation_report_{timestamp}.md"

    _write_report_csv(report_csv_path, report_rows)
    report_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_markdown_report(
        report_md_path,
        summary,
        {
            "samples": num_samples,
            "split": split,
            "strategies": list(strategy_configs.keys()),
        },
    )

    return {
        "summary": summary,
        "reports": {
            "csv": str(report_csv_path),
            "json": str(report_json_path),
            "markdown": str(report_md_path),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Critic-CoT evaluation on GSM8K.")
    parser.add_argument("--samples", type=int, default=settings.DEFAULT_EVAL_SAMPLES, help="Number of GSM8K samples.")
    parser.add_argument("--split", default=settings.DEFAULT_DATASET_SPLIT, choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the selected split before limiting.")
    parser.add_argument("--seed", type=int, default=settings.DEFAULT_SAMPLE_SEED, help="Random seed used for shuffling.")
    parser.add_argument(
        "--strategies",
        default=",".join(settings.STRATEGIES),
        help="Comma-separated strategy list: baseline,iter_refine,filter,majority",
    )
    parser.add_argument("--max-iterations", type=int, default=settings.DEFAULT_MAX_ITERATIONS, help="Iterations for iter_refine.")
    parser.add_argument("--filter-samples", type=int, default=settings.DEFAULT_FILTER_SAMPLES, help="Candidates for filter strategy.")
    parser.add_argument("--majority-samples", type=int, default=settings.DEFAULT_MAJORITY_SAMPLES, help="Candidates for majority strategy.")
    args = parser.parse_args()

    result = run_scaled_evaluation(
        num_samples=args.samples,
        split=args.split,
        shuffle=args.shuffle,
        seed=args.seed,
        strategies=args.strategies,
        max_iterations=args.max_iterations,
        filter_samples=args.filter_samples,
        majority_samples=args.majority_samples,
    )
    print(json.dumps(result, indent=2))
