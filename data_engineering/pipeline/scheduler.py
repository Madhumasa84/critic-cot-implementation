"""
Simple scheduler for recurring Critic-CoT evaluations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_engineering.config import settings
from data_engineering.pipeline.simple_pipeline import CriticCoTPipeline


def ensure_api_key_configured() -> None:
    if settings.has_configured_api_key():
        return
    raise RuntimeError(
        "OPENROUTER_API_KEY is not configured. Add it to config.py, .env, or your environment variables before starting the scheduler."
    )


class CriticCoTScheduler:
    """Run daily or interval-based Critic-CoT evaluations and log the results."""

    def __init__(self):
        self.pipeline = CriticCoTPipeline()
        self.log_path = settings.DAILY_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def run_daily_evaluation(
        self,
        samples: int = settings.DEFAULT_EVAL_SAMPLES,
        split: str = settings.DEFAULT_DATASET_SPLIT,
        strategy_configs: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Dict[str, Dict[str, object]]:
        ensure_api_key_configured()
        dataset = self.pipeline.ingestor.load_gsm8k(split=split, limit=samples)
        summary = self.pipeline.run_all_strategies(dataset, strategy_configs=strategy_configs)
        self._append_log(summary, sample_count=samples, split=split)
        return summary

    def run_once(
        self,
        samples: int = settings.DEFAULT_EVAL_SAMPLES,
        split: str = settings.DEFAULT_DATASET_SPLIT,
        strategy_configs: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Dict[str, Dict[str, object]]:
        return self.run_daily_evaluation(samples=samples, split=split, strategy_configs=strategy_configs)

    def run_continuous(
        self,
        samples: int = settings.DEFAULT_EVAL_SAMPLES,
        split: str = settings.DEFAULT_DATASET_SPLIT,
        strategy_configs: Optional[Dict[str, Dict[str, int]]] = None,
        daily_time: str = settings.DEFAULT_SCHEDULER_DAILY_TIME,
        interval_hours: Optional[int] = None,
    ) -> None:
        while True:
            now = datetime.now()
            next_run = self._next_run_time(now, daily_time=daily_time, interval_hours=interval_hours)
            sleep_seconds = max(int((next_run - now).total_seconds()), 0)
            print(f"Next evaluation scheduled for {next_run.isoformat()}")

            while sleep_seconds > 0:
                time.sleep(min(settings.CONTINUOUS_SLEEP_SECONDS, sleep_seconds))
                sleep_seconds -= settings.CONTINUOUS_SLEEP_SECONDS

            print(f"Starting scheduled evaluation at {datetime.now().isoformat()}")
            result = self.run_daily_evaluation(
                samples=samples,
                split=split,
                strategy_configs=strategy_configs,
            )
            print(json.dumps(result, indent=2))

    def _next_run_time(
        self,
        now: datetime,
        daily_time: str,
        interval_hours: Optional[int],
    ) -> datetime:
        if interval_hours:
            return now + timedelta(hours=interval_hours)

        hour_text, minute_text = daily_time.split(":")
        candidate = now.replace(
            hour=int(hour_text),
            minute=int(minute_text),
            second=0,
            microsecond=0,
        )
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    def _append_log(
        self,
        summary: Dict[str, Dict[str, object]],
        sample_count: int,
        split: str,
    ) -> str:
        row = {
            "timestamp": datetime.now().isoformat(),
            "samples": sample_count,
            "split": split,
        }

        for strategy in settings.STRATEGIES:
            if strategy not in summary:
                continue
            item = summary[strategy]
            row[f"acc_{strategy}"] = item["accuracy_pct"]
            row[f"latency_{strategy}"] = item["avg_latency_ms"]
            row[f"cost_{strategy}"] = item["total_cost_usd"]

        fieldnames = ["timestamp", "samples", "split"]
        for strategy in settings.STRATEGIES:
            fieldnames.extend(
                [
                    f"acc_{strategy}",
                    f"latency_{strategy}",
                    f"cost_{strategy}",
                ]
            )
        write_header = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return str(self.log_path)


def _build_strategy_configs(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    requested = [item.strip() for item in args.strategies.split(",") if item.strip()]
    configs: Dict[str, Dict[str, int]] = {}

    for strategy in requested:
        if strategy not in settings.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        if strategy == "iter_refine":
            configs[strategy] = {"max_iterations": args.max_iterations}
        elif strategy == "filter":
            configs[strategy] = {"num_samples": args.filter_samples}
        elif strategy == "majority":
            configs[strategy] = {"num_samples": args.majority_samples}
        else:
            configs[strategy] = {}
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schedule Critic-CoT evaluations.")
    parser.add_argument("--mode", choices=["once", "continuous"], default="once", help="Run a single evaluation or keep scheduling.")
    parser.add_argument("--samples", type=int, default=settings.DEFAULT_EVAL_SAMPLES, help="Number of GSM8K samples per run.")
    parser.add_argument("--split", default=settings.DEFAULT_DATASET_SPLIT, choices=["train", "test"], help="Dataset split.")
    parser.add_argument(
        "--strategies",
        default=",".join(settings.STRATEGIES),
        help="Comma-separated strategies to run.",
    )
    parser.add_argument("--max-iterations", type=int, default=settings.DEFAULT_MAX_ITERATIONS, help="Iterations for iter_refine.")
    parser.add_argument("--filter-samples", type=int, default=settings.DEFAULT_FILTER_SAMPLES, help="Candidates for filter strategy.")
    parser.add_argument("--majority-samples", type=int, default=settings.DEFAULT_MAJORITY_SAMPLES, help="Candidates for majority strategy.")
    parser.add_argument("--time", default=settings.DEFAULT_SCHEDULER_DAILY_TIME, help="Daily execution time in HH:MM format.")
    parser.add_argument("--interval-hours", type=int, default=None, help="Run continuously every N hours instead of at a daily clock time.")
    args = parser.parse_args()

    scheduler = CriticCoTScheduler()
    strategy_configs = _build_strategy_configs(args)

    if args.mode == "once":
        result = scheduler.run_once(samples=args.samples, split=args.split, strategy_configs=strategy_configs)
        print(json.dumps(result, indent=2))
    else:
        scheduler.run_continuous(
            samples=args.samples,
            split=args.split,
            strategy_configs=strategy_configs,
            daily_time=args.time,
            interval_hours=args.interval_hours,
        )
