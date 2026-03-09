"""
models/llm_model.py
─────────────────────────────────────────────────────────────
LLM abstraction layer.

Provides a unified interface for multiple LLM backends:
    • OpenAI    (gpt-4o-mini, gpt-4o, …)
    • Ollama    (llama3, mistral, … — local)
    • HuggingFace Inference API

All backends implement the same ``LLMBase`` interface:
    * ``generate(prompt, system_prompt, history)``      → str
    * ``agenerate(...)``                                → str  (async)
    * ``stream(prompt, system_prompt, history)``        → Iterator[str]

Supports prompt templates with Jinja2-style {variable} substitution.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Iterator

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from utils.logger import get_logger, log_llm_call

log = get_logger(__name__)


# ─── Prompt Templates ────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a knowledgeable AI assistant for a company's internal knowledge base.
Your role is to answer employee questions accurately using ONLY the provided document context.

Guidelines:
- Answer based ONLY on the provided context. Do not use outside knowledge.
- Always cite your sources using [1], [2], etc. corresponding to the context chunks provided.
- If the context does not contain enough information to answer the question, say so explicitly.
- Be concise but thorough. Use bullet points for lists.
- For policies, always quote exact numbers (days, amounts, percentages) when available.
- If multiple documents contain relevant information, synthesize them coherently.

Format your answer as:
1. Direct answer to the question
2. Supporting details (with citations)
3. Any caveats or limitations

Remember: You MUST cite sources. Every factual claim needs a [n] reference."""

RAG_USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

---

QUESTION: {question}

Please answer the question based on the context above. Include citations like [1], [2] etc."""

QUERY_REWRITE_PROMPT = """Given the conversation history and the follow-up question, 
rewrite the follow-up question to be a standalone question that captures the full intent.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


# ─── Base Interface ──────────────────────────────────────────
class LLMBase(ABC):
    """Abstract base class for all LLM backends."""

    def __init__(self, model: str, max_tokens: int = 1024, temperature: float = 0.1) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a response for the given prompt."""

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Async version — runs generate in a thread by default."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, system_prompt, history),
        )

    def stream(
        self,
        prompt: str,
        system_prompt: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        """Stream response tokens. Default: yield full response in one shot."""
        yield self.generate(prompt, system_prompt, history)

    def format_prompt(self, template: str, **kwargs: Any) -> str:
        """Simple {variable} substitution in prompt templates."""
        return template.format(**kwargs)


# ─── OpenAI Backend ──────────────────────────────────────────
class OpenAILLM(LLMBase):
    """
    OpenAI chat completion backend.

    Requires ``OPENAI_API_KEY`` in environment.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        super().__init__(model, max_tokens, temperature)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        self._client = OpenAI(api_key=api_key or None)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> str:
        messages = _build_messages(system_prompt, history, prompt)
        t0 = time.perf_counter()

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        usage = response.usage
        log_llm_call(
            provider="openai",
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed,
        )
        return response.choices[0].message.content or ""

    def stream(
        self,
        prompt: str,
        system_prompt: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        messages = _build_messages(system_prompt, history, prompt)
        with self._client.chat.completions.stream(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ) as stream:
            for text in stream.text_stream:
                yield text


# ─── Ollama Backend ──────────────────────────────────────────
class OllamaLLM(LLMBase):
    """
    Ollama local LLM backend.

    Requires an Ollama server running at ``base_url``.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        super().__init__(model, max_tokens, temperature)
        self.base_url = base_url
        try:
            import ollama as _ollama
            self._ollama = _ollama
        except ImportError as exc:
            raise ImportError("pip install ollama") from exc

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> str:
        messages = _build_messages(system_prompt, history, prompt)
        t0 = time.perf_counter()
        try:
            response = self._ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            elapsed = (time.perf_counter() - t0) * 1000
            log_llm_call(
                provider="ollama",
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed,
            )
            return response["message"]["content"]
        except Exception as exc:
            log.error(f"Ollama generation failed: {exc}")
            raise

    def stream(
        self,
        prompt: str,
        system_prompt: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        messages = _build_messages(system_prompt, history, prompt)
        for chunk in self._ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        ):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content


# ─── HuggingFace Backend ─────────────────────────────────────
class HuggingFaceLLM(LLMBase):
    """
    HuggingFace Inference API backend (hosted models).
    """

    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        api_token: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        super().__init__(model, max_tokens, temperature)
        try:
            import httpx
            self._httpx = httpx
        except ImportError as exc:
            raise ImportError("pip install httpx") from exc

        self.api_token = api_token
        self._api_url = f"https://api-inference.huggingface.co/models/{model}"
        self._headers = {"Authorization": f"Bearer {api_token}"}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> str:
        full_prompt = _format_instruct_prompt(system_prompt, history, prompt)
        t0 = time.perf_counter()
        response = self._httpx.post(
            self._api_url,
            headers=self._headers,
            json={
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "return_full_text": False,
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        elapsed = (time.perf_counter() - t0) * 1000
        log_llm_call("huggingface", self.model, 0, 0, elapsed)
        data = response.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
        return ""


# ─── LLM Factory ─────────────────────────────────────────────
def build_llm(
    provider: str,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    api_token: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> LLMBase:
    """
    Build and return the appropriate LLM backend.

    Parameters
    ----------
    provider    : "openai" | "ollama" | "huggingface"
    model       : Model name / identifier
    api_key     : OpenAI API key
    base_url    : Ollama server URL
    api_token   : HuggingFace API token
    max_tokens  : Maximum generation length
    temperature : Sampling temperature (lower = more deterministic)
    """
    provider = provider.lower()
    if provider == "openai":
        return OpenAILLM(
            model=model or "gpt-4o-mini",
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif provider == "ollama":
        return OllamaLLM(
            model=model or "llama3.2",
            base_url=base_url or "http://localhost:11434",
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif provider == "huggingface":
        return HuggingFaceLLM(
            model=model or "mistralai/Mistral-7B-Instruct-v0.2",
            api_token=api_token,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            "Choose: openai | ollama | huggingface"
        )


# ─── Private helpers ─────────────────────────────────────────
def _build_messages(
    system_prompt: str,
    history: list[dict[str, str]] | None,
    user_prompt: str,
) -> list[dict[str, str]]:
    """Assemble the message list for OpenAI-style chat APIs."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _format_instruct_prompt(
    system_prompt: str,
    history: list[dict[str, str]] | None,
    user_prompt: str,
) -> str:
    """Format a prompt string for instruction-tuned models (Mistral style)."""
    parts: list[str] = []
    if system_prompt:
        parts.append(f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n")
    if history:
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                parts.append(f"{content}")
    parts.append(f"[INST] {user_prompt} [/INST]")
    return "".join(parts)
