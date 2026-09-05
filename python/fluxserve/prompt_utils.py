# Copyright (c) 2026 FLUX-OSS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
    Prompt rendering shared by offline benchmarking and online serving.
"""


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
