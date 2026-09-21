"""Prompt rendering shared by offline benchmarking and online serving."""


def render_openai_messages(messages) -> str:
    """Render OpenAI-style messages using the format expected by LLaDA 2."""
    rendered = []
    for message in messages:
        role = message.get("role", "").upper()
        content = message.get("content", "")
        if role == "SYSTEM":
            rendered.append(f"<role>SYSTEM</role>{content}<|role_end|>")
        elif role == "ASSISTANT":
            rendered.append(f"<role>ASSISTANT</role>{content}<|role_end|>")
        else:
            rendered.append(f"<role>HUMAN</role>{content}<|role_end|>")
    rendered.append("<role>ASSISTANT</role>")
    return "".join(rendered)
