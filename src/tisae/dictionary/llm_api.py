"""
LLM API clients for latent dictionary labeling and verbalization judging.

Provides Claude (Anthropic) and OpenAI clients with retry, metadata logging,
and checkpoint/resume support via LLMLabeler.

SDKs are optional — missing SDK raises a clear error at construction, not import.

Usage:
    # Label dictionary
    python -m tisae.dictionary.llm_api \
        --experiment_dir experiments/tisae/my_exp \
        --provider claude --model claude-sonnet-4-6

    # Or with OpenAI
    python -m tisae.dictionary.llm_api \
        --experiment_dir experiments/tisae/my_exp \
        --provider openai --model gpt-4o
"""

from __future__ import annotations

import abc
import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .llm_labeler import LLMLabeler, SYSTEM_MESSAGE

logger = logging.getLogger(__name__)


@dataclass
class LLMCallMetadata:
    """Per-call metadata for reproducibility auditing."""
    provider: str
    model: str
    temperature: float
    timestamp: str
    latent_key: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


class LLMClient(abc.ABC):
    """Abstract base for LLM API clients."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    @property
    @abc.abstractmethod
    def provider(self) -> str: ...

    @abc.abstractmethod
    def _call_api(self, prompt: str, system_msg: str) -> Tuple[str, int, int]:
        """Raw API call. Returns (response_text, input_tokens, output_tokens)."""
        ...

    def label_one(
        self, prompt: str, system_msg: str, latent_key: str = "",
    ) -> Tuple[str, LLMCallMetadata]:
        """Send one prompt with retry. Returns (response_text, metadata)."""
        delay = self.retry_base_delay
        for attempt in range(self.max_retries):
            try:
                t0 = time.monotonic()
                text, in_tok, out_tok = self._call_api(prompt, system_msg)
                latency = (time.monotonic() - t0) * 1000
                meta = LLMCallMetadata(
                    provider=self.provider,
                    model=self.model,
                    temperature=self.temperature,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    latent_key=latent_key,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=round(latency, 1),
                )
                return text, meta
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(
                    f"API call failed for {latent_key} (attempt {attempt + 1}/{self.max_retries}): "
                    f"{e}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay = min(delay * 2, self.retry_max_delay)
        raise RuntimeError("unreachable")


class ClaudeClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic SDK not installed. Install with: pip install anthropic"
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def provider(self) -> str:
        return "anthropic"

    def _call_api(self, prompt: str, system_msg: str) -> Tuple[str, int, int]:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}],
        )
        return (
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai SDK not installed. Install with: pip install openai"
            )
        self._client = openai.OpenAI(api_key=api_key)

    @property
    def provider(self) -> str:
        return "openai"

    def _call_api(self, prompt: str, system_msg: str) -> Tuple[str, int, int]:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
        )
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )


def build_client(
    provider: str = "claude",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Factory for LLM clients."""
    if provider == "claude":
        return ClaudeClient(model=model or "claude-sonnet-4-6", api_key=api_key, **kwargs)
    elif provider == "openai":
        return OpenAIClient(model=model or "gpt-4o", api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'claude' or 'openai'.")


def label_dictionary(
    experiment_dir: str,
    client: LLMClient,
    system_msg: str = SYSTEM_MESSAGE,
    dry_run: bool = False,
) -> dict:
    """Label all unlabeled latents in a dictionary, with checkpoint/resume.

    Returns status dict with counts.
    """
    labeler = LLMLabeler(experiment_dir)
    unlabeled = labeler.get_unlabeled()
    log_path = Path(experiment_dir) / "latent_dictionary" / "llm_call_log.jsonl"

    print(f"Labeling status: {labeler.status()}")
    print(f"Provider: {client.provider}, model: {client.model}, "
          f"temperature: {client.temperature}")

    if not unlabeled:
        print("All latents already labeled.")
        return labeler.status()

    print(f"{len(unlabeled)} latents to label.")
    if dry_run:
        for k, c in unlabeled[:5]:
            print(f"  Would label k={k}, c={c}")
        if len(unlabeled) > 5:
            print(f"  ... and {len(unlabeled) - 5} more")
        return labeler.status()

    n_labeled = 0
    n_errors = 0
    for i, (k, c) in enumerate(unlabeled):
        prompt = labeler.read_prompt(k, c)
        latent_key = f"{k}_{c}"

        try:
            response_text, meta = client.label_one(prompt, system_msg, latent_key)
            label = labeler.ingest_response(k, c, response_text)

            with open(log_path, "a") as f:
                f.write(json.dumps(asdict(meta)) + "\n")

            status = "OK" if label.get("clinical_label") != "PARSE_ERROR" else "PARSE_ERROR"
            if status == "PARSE_ERROR":
                n_errors += 1
            else:
                n_labeled += 1

            print(f"  [{i + 1}/{len(unlabeled)}] k={k},c={c}: "
                  f"{label.get('clinical_label', '?')[:50]} [{status}]")

        except Exception as e:
            n_errors += 1
            logger.error(f"Failed to label k={k},c={c}: {e}")
            print(f"  [{i + 1}/{len(unlabeled)}] k={k},c={c}: FAILED ({e})")

    labeler.merge_into_dictionary()

    final_status = labeler.status()
    print(f"\nDone. Labeled: {n_labeled}, errors: {n_errors}")
    print(f"Final status: {final_status}")
    return final_status


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM labeling for TISAE latent dictionary")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="TISAE experiment directory (containing latent_dictionary/)")
    parser.add_argument("--provider", type=str, choices=["claude", "openai"],
                        default="claude")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: claude-sonnet-4-6 / gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (default: from ANTHROPIC_API_KEY / OPENAI_API_KEY env)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be labeled without calling API")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    client = build_client(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )
    label_dictionary(
        experiment_dir=args.experiment_dir,
        client=client,
        dry_run=args.dry_run,
    )
