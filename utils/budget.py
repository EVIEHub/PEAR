"""Compute-budget tracker with fail-fast semantics.

Per ExpPlan.md section 2.3 ("Fairness enforcement"), every condition must use
the same total number of model calls and the same maximum tokens per call.
This module exposes a small ``Budget`` dataclass that runs as a counter; the
:meth:`Budget.charge` method raises :class:`BudgetExceededError` if the
caller would exceed the configured cap, which lets the experiment runner
treat a budget violation as a hard error rather than as silent extra spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


class BudgetExceededError(RuntimeError):
    """Raised when a node would exceed the configured compute budget."""


@dataclass
class Budget:
    """Mutable counter for model calls and tokens.

    Attributes
    ----------
    max_calls:
        Hard cap on total model calls (agents + judge). ``0`` disables.
    max_tokens:
        Hard cap on total tokens consumed across the run. ``0`` disables.
    calls:
        Live count of model calls so far.
    prompt_tokens, completion_tokens:
        Token counters; updated by the model wrapper when token usage is
        reported.
    """

    max_calls: int = 0
    max_tokens: int = 0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extras: Dict[str, int] = field(default_factory=dict)

    def charge(self, *, calls: int = 1, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Account for one model call. Raises if this would exceed any cap.

        Parameters
        ----------
        calls:
            Number of model calls being charged (usually 1).
        prompt_tokens:
            Tokens in the prompt(s) that were sent.
        completion_tokens:
            Tokens in the completion that was returned.
        """
        new_calls = self.calls + calls
        new_tokens = self.prompt_tokens + prompt_tokens + self.completion_tokens + completion_tokens

        if self.max_calls and new_calls > self.max_calls:
            raise BudgetExceededError(
                f"Would exceed max_calls cap: {new_calls} > {self.max_calls}"
            )
        if self.max_tokens and new_tokens > self.max_tokens:
            raise BudgetExceededError(
                f"Would exceed max_tokens cap: {new_tokens} > {self.max_tokens}"
            )

        self.calls = new_calls
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the current counters."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "max_calls": self.max_calls,
            "max_tokens": self.max_tokens,
            "extras": dict(self.extras),
        }
