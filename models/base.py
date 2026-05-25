"""Abstract base class for PEAR model backends.

The whole experiment harness only ever calls a single method on a model:
``generate(prompt, *, max_tokens, temperature, system, agent_id)``. Sub-
classes are free to do whatever they like internally as long as they return a
:class:`Generation` with the produced text and (optionally) token counts.

This deliberate minimalism matches ExpPlan.md's "swap only the topology
scheduler" principle: keep the LLM interface fixed across baselines and PEAR
so that comparisons remain apples-to-apples.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class Generation:
    """One model completion plus optional usage metadata.

    Attributes
    ----------
    text:
        The completion string (already stripped of any provider-specific
        wrapping such as ``"<|assistant|>"`` markers).
    prompt_tokens, completion_tokens:
        Best-effort token counts. Set to ``0`` if the backend cannot report
        them (this is the case for the stub and for Ollama in some setups).
    raw:
        Provider-native response object, kept for debugging. Never written to
        the JSONL trace file.
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: object = field(default=None, repr=False)


class BaseLLM(abc.ABC):
    """Common interface for every LLM backend used in the experiments.

    Implementations should be *stateless across calls*: any caching should be
    explicit (e.g. a HuggingFace KV cache) and should not leak between
    different agents in the same debate, since that would defeat the
    fairness controls described in ExpPlan.md.
    """

    #: Free-form name shown in logs (set by the factory from the registry key).
    name: str = "BaseLLM"

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        system: Optional[str] = None,
        agent_id: Optional[int] = None,
    ) -> Generation:
        """Produce one completion for ``prompt``.

        Parameters
        ----------
        prompt:
            User-side prompt. Should not include any system instructions; pass
            those via the ``system`` argument so providers that natively
            distinguish system messages can use them properly.
        max_tokens:
            Maximum tokens to generate.
        temperature:
            Sampling temperature.
        system:
            Optional system instruction (role / persona / format).
        agent_id:
            Optional agent identifier; backends that route per-agent (e.g.
            different personas) can use this to pick parameters.

        Returns
        -------
        Generation
        """
        raise NotImplementedError

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        system: Optional[str] = None,
        agent_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> list[Generation]:
        """Produce one completion per prompt.

        Backends can override this for true batched inference. The default
        preserves existing semantics by dispatching sequential single calls.
        """
        if agent_ids is None:
            agent_ids = [None] * len(prompts)
        if len(agent_ids) != len(prompts):
            raise ValueError("agent_ids must have the same length as prompts")
        return [
            self.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                agent_id=agent_id,
            )
            for prompt, agent_id in zip(prompts, agent_ids)
        ]

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r}>"
