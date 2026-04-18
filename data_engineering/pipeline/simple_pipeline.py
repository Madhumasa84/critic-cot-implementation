"""
Main execution pipeline for Critic-CoT evaluation.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_engineering.config import settings
from data_engineering.pipeline.critic_cot_wrapper import execute_strategy
from data_engineering.pipeline.data_ingestion import DataIngestion
from data_engineering.storage.reasoning_db import ReasoningTraceDB


class CriticCoTPipeline:
    """Run Critic-CoT strategies on standardized samples and persist outputs."""

    def __init__(self, db_path: Optional[str] = None):
        self.db = ReasoningTraceDB(db_path or str(settings.DATABASE_PATH))
        self.ingestor = DataIngestion(cache_dir=str(settings.CACHE_DIR))
        self.results: List[Dict[str, Any]] = []
        self.strategy_results: Dict[str, List[Dict[str, Any]]] = {}

    def run_on_samples(
        self,
        samples: List[Dict[str, Any]],
        strategy: str = "baseline",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        standardized_samples = self.ingestor.format_for_pipeline(samples)
        run_rows: List[Dict[str, Any]] = []

        for sample in standardized_samples:
            trace = execute_strategy(
                question=sample["question"],
                expected_answer=sample["answer"],
                strategy_name=strategy,
                sample_metadata=sample,
                max_iterations=kwargs.get("max_iterations", settings.DEFAULT_MAX_ITERATIONS),
                num_samples=kwargs.get("num_samples", settings.DEFAULT_MAJORITY_SAMPLES),
                model=kwargs.get("model"),
            )

            self.db.save_trace(trace)
            row = self._result_row_from_trace(trace)
            run_rows.append(row)

        self.results = run_rows
        self.strategy_results[strategy] = run_rows
        return self._summarize_results(strategy, run_rows, kwargs)

    def run_all_strategies(
        self,
        samples: List[Dict[str, Any]],
        strategy_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        configs = strategy_configs or {
            "baseline": {},
            "iter_refine": {"max_iterations": settings.DEFAULT_MAX_ITERATIONS},
            "filter": {"num_samples": settings.DEFAULT_FILTER_SAMPLES},
            "majority": {"num_samples": settings.DEFAULT_MAJORITY_SAMPLES},
        }

        summaries: Dict[str, Dict[str, Any]] = {}

        for strategy_name, params in configs.items():
            summary = self.run_on_samples(samples, strategy=strategy_name, **params)
            summaries[strategy_name] = summary
            self._save_results_csv(
                settings.DATA_DIR / f"{strategy_name}_results.csv",
                self.strategy_results[strategy_name],
            )

        self.db.update_daily_metrics()
        exported_tables = self.db.export_all_to_csv(str(settings.EXPORT_DIR))
        report_json_path = self._save_summary_json(summaries)

        return {
            **summaries,
            "_artifacts": {
                "database": str(self.db.db_path),
                "exports": exported_tables,
                "summary_json": report_json_path,
            },
        }

    def _result_row_from_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        metadata = trace.get("metadata", {})
        return {
            "trace_id": trace.get("trace_id"),
            "timestamp": trace.get("timestamp"),
            "sample_id": trace.get("question", {}).get("id"),
            "source": trace.get("question", {}).get("source"),
            "split": trace.get("question", {}).get("split"),
            "strategy": trace.get("strategy", {}).get("name"),
            "predicted": trace.get("final_answer"),
            "expected": trace.get("question", {}).get("expected_answer"),
            "correct": bool(trace.get("is_correct")),
            "latency_ms": metadata.get("api_latency_ms", 0.0),
            "total_tokens": metadata.get("total_tokens", 0),
            "cost_usd": metadata.get("cost_usd", 0.0),
            "iterations": len(trace.get("iterations", [])),
            "steps": metadata.get("total_steps", 0),
            "error": metadata.get("error"),
        }

    def _summarize_results(
        self,
        strategy: str,
        rows: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        total = len(rows)
        correct = sum(1 for row in rows if row["correct"])
        accuracy_pct = round((correct / total) * 100, 2) if total else 0.0
        avg_latency = round(mean(row["latency_ms"] for row in rows), 2) if rows else 0.0
        avg_tokens = round(mean(row["total_tokens"] for row in rows), 2) if rows else 0.0
        total_cost = round(sum(row["cost_usd"] for row in rows), 6)

        return {
            "strategy": strategy,
            "parameters": params,
            "total_samples": total,
            "correct_count": correct,
            "accuracy_pct": accuracy_pct,
            "avg_latency_ms": avg_latency,
            "avg_tokens": avg_tokens,
            "total_cost_usd": total_cost,
            "result_csv": str(settings.DATA_DIR / f"{strategy}_results.csv"),
        }

    def _save_results_csv(self, path: Path, rows: List[Dict[str, Any]]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
        return str(path)

    def _save_summary_json(self, summaries: Dict[str, Dict[str, Any]]) -> str:
        path = settings.REPORT_DIR / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(summaries, indent=2, ensure_ascii=True), encoding="utf-8")
        return str(path)


if __name__ == "__main__":
    pipeline = CriticCoTPipeline()
    gsm8k_samples = pipeline.ingestor.load_gsm8k(limit=5)
    summary = pipeline.run_all_strategies(gsm8k_samples)
    print(json.dumps(summary, indent=2))
