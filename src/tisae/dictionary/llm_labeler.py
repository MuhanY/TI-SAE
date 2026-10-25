"""
LLM Labeler for Latent Dictionary.

Handles prompt discovery, JSON response parsing, resume/checkpoint,
merging labels into the latent dictionary, and labeled summary generation.

Usage:
    labeler = LLMLabeler(experiment_dir)

    # Check what needs labeling
    unlabeled = labeler.get_unlabeled()

    # Ingest a response
    labeler.ingest_response(k=5, c=0, response_text='{"clinical_label": ...}')

    # Merge labels into the main dictionary JSON
    labeler.merge_into_dictionary()

    # Generate labeled summary markdown
    labeler.generate_labeled_summary()
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Expected fields in the LLM response JSON
EXPECTED_FIELDS = [
    "clinical_label",
    "clinical_description",
]

SYSTEM_MESSAGE = (
    "You are an expert clinical informaticist interpreting Temporal Island "
    "Sparse Autoencoder (TI-SAE) latents from ICU data (MIMIC-III). Analyze "
    "the provided data and respond with ONLY a JSON object as instructed in "
    "the prompt. No markdown fences, no extra text."
)


def parse_llm_json(response_text: str) -> dict:
    """
    Parse JSON from LLM response with 3-level fallback.

    1. Direct json.loads
    2. Extract from ```json ... ``` fences
    3. Greedy regex for outermost { ... }
    4. Return PARSE_ERROR sentinel
    """
    text = response_text.strip()

    # Level 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return _validate_fields(result)
    except json.JSONDecodeError:
        pass

    # Level 2: fenced JSON block
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1))
            if isinstance(result, dict):
                return _validate_fields(result)
        except json.JSONDecodeError:
            pass

    # Level 3: greedy outermost braces
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return _validate_fields(result)
        except json.JSONDecodeError:
            pass

    # All failed
    return {
        "clinical_label": "PARSE_ERROR",
        "clinical_description": "",
        "_raw_response": text[:2000],
    }


def _validate_fields(result: dict) -> dict:
    """Ensure all expected fields exist, fill missing with empty string."""
    for field in EXPECTED_FIELDS:
        if field not in result:
            result[field] = ""
    return result


def make_verbalizer_card(entry: dict) -> str:
    """
    Compact per-latent summary for the verbalizer LLM.

    ~150-250 chars per latent. Includes label, first sentence of description,
    key variable names, mortality OR, and activation frequency.

    Format:
        [k=42,c=3] Progressive AKI (OR=3.21, freq=8.2%)
        Rising creatinine and BUN suggesting acute kidney injury.
        Key vars: Creatinine, BUN, Urine Output
    """
    k = entry.get("latent_k", "?")
    c = entry.get("channel_c", "?")
    label = entry.get("clinical_label", "")
    if not label:
        label = f"k={k}_c={c}"

    odds = entry.get("mortality_odds_ratio", 1.0)
    freq = entry.get("activation_frequency", 0)

    # First sentence of clinical_description
    desc = entry.get("clinical_description", "")
    first_sentence = desc.split(". ")[0].rstrip(".") + "." if desc else ""

    # Key variable names from decoder composition
    var_comp = entry.get("variable_composition", [])
    key_vars_raw = ", ".join(v["var_name"] for v in var_comp[:5])

    parts = [f"[k={k},c={c}] {label} (OR={odds:.2f}, freq={freq*100:.1f}%)"]
    if first_sentence:
        parts.append(first_sentence)
    if key_vars_raw:
        parts.append(f"Key vars: {key_vars_raw}")

    return "\n".join(parts)


class LLMLabeler:
    """Manages LLM labeling state for a latent dictionary experiment."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.dict_dir = self.experiment_dir / "latent_dictionary"
        self.prompt_dir = self.dict_dir / "llm_prompts"
        self.response_dir = self.dict_dir / "llm_responses"
        self.labels_path = self.dict_dir / "llm_labels.json"
        self.dictionary_path = self.dict_dir / "latent_dictionary.json"

        if not self.prompt_dir.exists():
            raise FileNotFoundError(
                f"Prompt directory not found: {self.prompt_dir}\n"
                "Run the latent dictionary builder first."
            )

        self._prompts = self._discover_prompts()
        self._labels = self._load_labels()

    def _discover_prompts(self) -> Dict[Tuple[int, int], Path]:
        """Find all latent_k{K}_c{C}.txt prompt files."""
        prompts = {}
        pattern = re.compile(r"latent_k(\d+)_c(\d+)\.txt")
        for f in sorted(self.prompt_dir.glob("latent_k*_c*.txt")):
            m = pattern.match(f.name)
            if m:
                k, c = int(m.group(1)), int(m.group(2))
                prompts[(k, c)] = f
        return prompts

    def _load_labels(self) -> dict:
        """Load existing labels JSON. Format: {"k_c": {fields...}, ...}"""
        if self.labels_path.exists():
            with open(self.labels_path) as f:
                return json.load(f)
        return {}

    def _save_labels(self):
        """Save labels JSON to disk."""
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.labels_path, "w") as f:
            json.dump(self._labels, f, indent=2)

    @staticmethod
    def _key(k: int, c: int) -> str:
        return f"{k}_{c}"

    @staticmethod
    def _parse_key(key: str) -> Tuple[int, int]:
        parts = key.split("_")
        return int(parts[0]), int(parts[1])

    def get_unlabeled(self) -> List[Tuple[int, int]]:
        """Return (k, c) pairs that have prompts but no valid labels."""
        unlabeled = []
        for (k, c) in sorted(self._prompts.keys()):
            key = self._key(k, c)
            if key not in self._labels:
                unlabeled.append((k, c))
            elif self._labels[key].get("clinical_label") == "PARSE_ERROR":
                unlabeled.append((k, c))
        return unlabeled

    def get_labeled(self) -> List[Tuple[int, int]]:
        """Return (k, c) pairs that have valid labels."""
        labeled = []
        for (k, c) in sorted(self._prompts.keys()):
            key = self._key(k, c)
            if key in self._labels and self._labels[key].get("clinical_label") != "PARSE_ERROR":
                labeled.append((k, c))
        return labeled

    def read_prompt(self, k: int, c: int) -> str:
        """Read the prompt text for a (k, c) pair."""
        path = self._prompts.get((k, c))
        if path is None:
            raise KeyError(f"No prompt found for k={k}, c={c}")
        return path.read_text()

    def ingest_response(self, k: int, c: int, response_text: str) -> dict:
        """
        Parse an LLM response and save it.

        Saves raw response to llm_responses/ and parsed label to llm_labels.json.
        Returns the parsed label dict.
        """
        label = parse_llm_json(response_text)

        self.response_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self.response_dir / f"latent_k{k}_c{c}.json"
        with open(raw_path, "w") as f:
            json.dump(label, f, indent=2)

        key = self._key(k, c)
        self._labels[key] = label
        self._save_labels()

        return label

    def ingest_from_response_files(self) -> int:
        """
        Scan llm_responses/ directory and ingest any responses not yet in labels.

        Returns number of newly ingested responses.
        """
        if not self.response_dir.exists():
            return 0

        count = 0
        pattern = re.compile(r"latent_k(\d+)_c(\d+)\.json")
        for f in sorted(self.response_dir.glob("latent_k*_c*.json")):
            m = pattern.match(f.name)
            if not m:
                continue
            k, c = int(m.group(1)), int(m.group(2))
            key = self._key(k, c)

            # Skip if already labeled (and not PARSE_ERROR)
            if key in self._labels and self._labels[key].get("clinical_label") != "PARSE_ERROR":
                continue

            with open(f) as fh:
                data = json.load(fh)

            # Support two formats:
            # 1. {"raw_response": "<json string>"} -- from external tools
            # 2. Direct label JSON {"clinical_label": ...} -- from subagent Write
            raw = data.get("raw_response", "")
            if raw:
                label = parse_llm_json(raw)
            elif "clinical_label" in data:
                label = _validate_fields(data)
            else:
                continue
            self._labels[key] = label
            count += 1

        if count > 0:
            self._save_labels()
        return count

    def merge_into_dictionary(self) -> int:
        """
        Merge LLM labels into the main latent_dictionary.json.

        Adds/updates clinical_label, clinical_description, and verbalizer_card
        on each entry. Returns number of entries updated.
        """
        if not self.dictionary_path.exists():
            raise FileNotFoundError(f"Dictionary not found: {self.dictionary_path}")

        with open(self.dictionary_path) as f:
            dictionary = json.load(f)

        count = 0
        for entry in dictionary:
            k, c = entry["latent_k"], entry["channel_c"]
            key = self._key(k, c)
            if key in self._labels:
                label = self._labels[key]
                for field in EXPECTED_FIELDS:
                    entry[field] = label.get(field, "")
                entry["verbalizer_card"] = make_verbalizer_card(entry)
                count += 1

        with open(self.dictionary_path, "w") as f:
            json.dump(dictionary, f, indent=2)

        print(f"Merged {count} labels into {self.dictionary_path}")
        return count

    def generate_labeled_summary(self, output_path: Optional[str] = None) -> str:
        """
        Generate a markdown summary table with clinical labels.

        Returns the markdown string and saves to file.
        """
        if not self.dictionary_path.exists():
            raise FileNotFoundError(f"Dictionary not found: {self.dictionary_path}")

        with open(self.dictionary_path) as f:
            dictionary = json.load(f)

        lines = ["# Latent Dictionary -- LLM-Labeled Summary", ""]

        n_labeled = sum(1 for e in dictionary if e.get("clinical_label"))
        lines.append(f"Total entries: {len(dictionary)} | Labeled: {n_labeled}")
        lines.append("")

        lines.append(
            "| k | c | Channel | Clinical Label | Top Variables "
            "| Freq | Mort OR |"
        )
        lines.append(
            "|---|---|---------|----------------|---------------"
            "|------|---------|"
        )

        for entry in sorted(
            dictionary, key=lambda e: e.get("mortality_odds_ratio", 0), reverse=True
        ):
            k = entry["latent_k"]
            c = entry["channel_c"]
            ch = entry.get("channel_name", "?")
            label = entry.get("clinical_label", "")
            vars_str = ", ".join(
                v["var_name"] for v in entry.get("variable_composition", [])[:3]
            )
            freq = f"{entry.get('activation_frequency', 0) * 100:.1f}%"
            odds = f"{entry.get('mortality_odds_ratio', 1.0):.2f}"

            if len(label) > 40:
                label = label[:37] + "..."

            lines.append(
                f"| {k} | {c} | {ch} | {label} | {vars_str} "
                f"| {freq} | {odds} |"
            )

        md = "\n".join(lines)

        if output_path is None:
            output_path = self.dict_dir / "latent_dictionary_labeled_summary.md"
        else:
            output_path = Path(output_path)

        with open(output_path, "w") as f:
            f.write(md)
        print(f"Saved labeled summary: {output_path}")

        return md

    def status(self) -> dict:
        """Return labeling status summary."""
        unlabeled = self.get_unlabeled()
        labeled = self.get_labeled()
        parse_errors = sum(
            1 for v in self._labels.values()
            if v.get("clinical_label") == "PARSE_ERROR"
        )
        return {
            "total_prompts": len(self._prompts),
            "labeled": len(labeled),
            "unlabeled": len(unlabeled),
            "parse_errors": parse_errors,
        }
