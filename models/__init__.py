"""LLM model framework.

The contract is intentionally tiny: every backend implements
:class:`models.base.BaseLLM` with a single ``generate`` method. The
factory in :mod:`models.factory` instantiates a backend from an entry
in ``configs/models.yaml`` (or an inline dict) and dispatches to
:mod:`models.model`, which holds the closed-source (OpenAI, Anthropic) and
open-source (Hugging Face Transformers, vLLM) backends in one module.

Provider names recognised by the factory: ``openai``, ``anthropic``,
``hf`` (transformers), ``vllm``.
"""

from models.base import BaseLLM, Generation
from models.factory import build_llm, load_model_registry

__all__ = [
    "BaseLLM",
    "Generation",
    "build_llm",
    "load_model_registry",
]
