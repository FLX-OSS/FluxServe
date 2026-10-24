import copy

from serve_smoke import completion_problem, served_model

# What fluxserve returned in CI run 40115244950: no `usage` block.
RESPONSE = {
    "id": "043e7ab13fe34b93be8006573aba3544",
    "object": "chat.completion",
    "model": "nvidia/Nemotron-Labs-Diffusion-14B",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "\n84"}, "finish_reason": "stop"}],
}


def test_completion_without_usage_passes():
    assert completion_problem(200, RESPONSE) == ("84", None)


def test_completion_problems():
    empty = copy.deepcopy(RESPONSE)
    empty["choices"][0]["message"]["content"] = "  "
    assert completion_problem(200, empty)[1] == "the completion is empty"
    assert completion_problem(500, RESPONSE)[1] == "HTTP 500"
    no_tokens = dict(RESPONSE, usage={"completion_tokens": 0})
    assert completion_problem(200, no_tokens)[1] == "the server reported 0 completion tokens"
    assert completion_problem(200, dict(RESPONSE, usage={"completion_tokens": 3}))[1] is None


def test_reasoning_content_counts_as_an_answer():
    reasoning = copy.deepcopy(RESPONSE)
    reasoning["choices"][0]["message"] = {"role": "assistant", "content": None, "reasoning_content": "12 * 7 = 84"}
    assert completion_problem(200, reasoning) == ("12 * 7 = 84", None)


def test_served_model_reads_the_launch_command():
    command = "fluxserve launch --model nvidia/Nemotron-Labs-Diffusion-14B --host 127.0.0.1 --port 18000"
    assert served_model(command) == "nvidia/Nemotron-Labs-Diffusion-14B"
