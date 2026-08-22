from fluxserve.prompt_utils import render_openai_messages
from fluxserve.backend.entrypoints.http_server import _messages_to_prompt


def test_render_openai_messages_matches_llada_offline_format():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "prior answer"},
    ]

    assert render_openai_messages(messages) == (
        "<role>SYSTEM</role>rules<|role_end|>"
        "<role>HUMAN</role>question<|role_end|>"
        "<role>ASSISTANT</role>prior answer<|role_end|>"
        "<role>ASSISTANT</role>"
    )


def test_online_prompt_uses_llada_renderer_by_default():
    messages = [{"role": "user", "content": "question"}]

    assert _messages_to_prompt(messages, object()) == render_openai_messages(messages)


def test_online_prompt_can_apply_tokenizer_template():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return "templated prompt"

    assert _messages_to_prompt(
        [{"role": "user", "content": "question"}],
        Tokenizer(),
        apply_template=True,
    ) == "templated prompt"
