from fluxserve.backend.entrypoints.prompt_utils import render_openai_messages
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


def test_online_prompt_forwards_chat_template_kwargs():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                                enable_thinking):
            return f"thinking={enable_thinking}"

    assert _messages_to_prompt(
        [{"role": "user", "content": "question"}],
        Tokenizer(),
        apply_template=True,
        template_kwargs={"enable_thinking": True},
    ) == "thinking=True"


def test_online_prompt_kwargs_do_not_collide_with_the_defaults():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return f"tokenize={tokenize} generation={add_generation_prompt}"

    render = lambda kwargs: _messages_to_prompt(
        [{"role": "user", "content": "question"}],
        Tokenizer(),
        apply_template=True,
        template_kwargs=kwargs,
    )
    assert render({"tokenize": False}) == "tokenize=False generation=True"
    assert render({"add_generation_prompt": True}) == "tokenize=False generation=True"
    assert render({"add_generation_prompt": False}) == "tokenize=False generation=False"
