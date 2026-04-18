"""
Dataset ingestion and normalization utilities for reasoning tasks.
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
except ImportError:  # pragma: no cover - optional dependency
    HFDataset = None
    load_dataset = None


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"


class DataIngestion:
    """Load, cache, and normalize reasoning datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_cache_dir = self.cache_dir / "hf"
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)

    def load_gsm8k(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        cache_key = self._cache_file_name("gsm8k", split, limit, shuffle, seed)
        cache_file = self.cache_dir / cache_key

        if use_cache and cache_file.exists():
            return self._read_json(cache_file)

        all_samples = self._load_gsm8k_from_local_arrow(split)
        if not all_samples:
            if load_dataset is None:
                fallback = self._load_cached_snapshot("gsm8k", split)
                if fallback:
                    return self._apply_limit_and_shuffle(fallback, limit, shuffle, seed)
                raise ImportError(
                    "The 'datasets' package is required to download GSM8K. "
                    "Install it or reuse an existing cached JSON snapshot."
                )

            dataset = load_dataset(
                "gsm8k",
                "main",
                split=split,
                cache_dir=str(self.hf_cache_dir),
            )

            all_samples = [
                self._format_gsm8k_record(index=index, item=item, split=split)
                for index, item in enumerate(dataset)
            ]

        self._write_json(self.cache_dir / f"gsm8k_{split}_snapshot.json", all_samples)

        samples = self._apply_limit_and_shuffle(all_samples, limit, shuffle, seed)
        self._write_json(cache_file, samples)
        return samples

    def format_for_pipeline(
        self,
        samples: Iterable[Dict[str, Any]],
        source: str = "custom",
        split: str = "user",
    ) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []

        for index, item in enumerate(samples):
            question = item.get("question") or item.get("text") or item.get("prompt") or ""
            answer = (
                item.get("answer")
                or item.get("expected_answer")
                or item.get("label")
                or item.get("target")
                or ""
            )

            formatted.append(
                {
                    "id": item.get("id", index),
                    "question": str(question).strip(),
                    "answer": self._extract_expected_answer(str(answer)),
                    "raw_answer": str(answer),
                    "source": item.get("source", source),
                    "split": item.get("split", split),
                    "metadata": item.get("metadata", {}),
                }
            )

        return formatted

    def save_samples(self, samples: List[Dict[str, Any]], filename: str) -> str:
        path = self.cache_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = sorted({key for sample in samples for key in sample.keys()})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)
        return str(path)

    def load_samples_from_csv(self, filename: str) -> List[Dict[str, Any]]:
        path = self.cache_dir / filename
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _format_gsm8k_record(
        self,
        index: int,
        item: Dict[str, Any],
        split: str,
    ) -> Dict[str, Any]:
        raw_answer = item["answer"]
        return {
            "id": index,
            "question": item["question"].strip(),
            "answer": self._extract_expected_answer(raw_answer),
            "raw_answer": raw_answer,
            "source": "gsm8k",
            "split": split,
            "metadata": {},
        }

    def _extract_expected_answer(self, answer_text: str) -> str:
        if "####" in answer_text:
            answer_text = answer_text.split("####")[-1]

        cleaned = answer_text.strip()
        cleaned = cleaned.replace("\\$", "$").replace("\\%", "%")
        cleaned = cleaned.replace(",", "")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _load_gsm8k_from_local_arrow(self, split: str) -> List[Dict[str, Any]]:
        if HFDataset is None:
            return []

        pattern = f"gsm8k-{split}.arrow"
        arrow_files = sorted(self.cache_dir.rglob(pattern))
        if not arrow_files:
            return []

        dataset = HFDataset.from_file(str(arrow_files[0]))
        return [
            self._format_gsm8k_record(index=index, item=item, split=split)
            for index, item in enumerate(dataset)
        ]

    def _apply_limit_and_shuffle(
        self,
        samples: List[Dict[str, Any]],
        limit: Optional[int],
        shuffle: bool,
        seed: int,
    ) -> List[Dict[str, Any]]:
        ordered = list(samples)
        if shuffle:
            random.Random(seed).shuffle(ordered)
        if limit is not None:
            ordered = ordered[:limit]
        return ordered

    def _cache_file_name(
        self,
        dataset_name: str,
        split: str,
        limit: Optional[int],
        shuffle: bool,
        seed: int,
    ) -> str:
        limit_token = str(limit) if limit is not None else "all"
        shuffle_token = f"shuffle_{seed}" if shuffle else "ordered"
        return f"{dataset_name}_{split}_{limit_token}_{shuffle_token}.json"

    def _load_cached_snapshot(self, dataset_name: str, split: str) -> List[Dict[str, Any]]:
        path = self.cache_dir / f"{dataset_name}_{split}_snapshot.json"
        if not path.exists():
            return []
        return self._read_json(path)

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _read_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    ingestor = DataIngestion()
    try:
        demo_samples = ingestor.load_gsm8k(limit=3)
        print(json.dumps(demo_samples, indent=2))
    except Exception as exc:  # pragma: no cover - manual usage
        print(f"Data ingestion check failed: {exc}")
