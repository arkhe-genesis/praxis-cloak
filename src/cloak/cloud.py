import os
import time

_client = None


def _get_client():
    global _client
    if _client is None:
        # Lazy import so --skip-cloud works without `anthropic` installed.
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running the harness."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def call_cloud(model: str, prompt: str, max_tokens: int = 1024) -> tuple[str, int]:
    """Send a single prompt to the cloud model. Returns (response_text, elapsed_ms)."""
    return call_cloud_chat(model, [{"role": "user", "content": prompt}], max_tokens)


def call_cloud_chat(model: str, messages: list[dict], max_tokens: int = 1024) -> tuple[str, int]:
    """Send a multi-turn message list to the cloud model. Returns (response_text, elapsed_ms).

    `messages` is the conversation as the provider sees it — already scrubbed.
    """
    client = _get_client()
    t0 = time.monotonic()
    resp = client.messages.create(model=model, max_tokens=max_tokens, messages=messages)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    return text, elapsed_ms
