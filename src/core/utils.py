"""
Lab 11 — Helper Utilities
"""
import asyncio
from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    Includes retry with backoff for transient 429 / 503 API spikes.
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        last_exc = None
        for attempt in range(5):
            try:
                text = await runner.chat(agent, user_message)
                return text, None
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                if any(code in err_str for code in ("429", "503", "502", "rate", "unavailable", "exhausted", "overloaded")) and attempt < 4:
                    await asyncio.sleep(3.0 * (attempt + 1))
                    continue
                raise
        raise last_exc

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    last_exc = None
    for attempt in range(6):
        session = None
        if session_id is not None and attempt == 0:
            try:
                session = await runner.session_service.get_session(
                    app_name=app_name, user_id=user_id, session_id=session_id
                )
            except (ValueError, KeyError):
                pass

        if session is None:
            try:
                session = await runner.session_service.create_session(
                    app_name=app_name, user_id=user_id
                )
            except Exception:
                session = await runner.session_service.create_session(
                    app_name=app_name, user_id=user_id
                )

        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )

        final_response = ""
        try:
            async for event in runner.run_async(
                user_id=user_id, session_id=session.id, new_message=content
            ):
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response += part.text
            return final_response, session
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            # Do not retry if daily quota is exhausted
            if "perday" in err_str or "per day" in err_str or "generate_content_free_tier_requests" in err_str:
                raise
            if any(code in err_str for code in ("429", "503", "502", "resource_exhausted", "unavailable", "high demand")) and attempt < 3:
                wait_s = 12.0 * (attempt + 1)
                print(f"  [Transient API {type(exc).__name__} — retrying in {wait_s:.0f}s (attempt {attempt + 1}/4)]")
                await asyncio.sleep(wait_s)
                continue
            raise

    raise last_exc
