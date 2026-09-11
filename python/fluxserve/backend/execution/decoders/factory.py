# Copyright (c) 2026 FLUX-OSS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from fluxserve.backend.execution.forward_batch_info import RunnerConfig
from fluxserve.backend.utils.server_args import ServerArgs

from .hierarchy import HierarchyDecoder
from .joint_threshold import JointThresholdDecoder
from .levenshtein import LevenshteinJointDecoder
from .threshold import CreditThresholdParallelDecoder, ThresholdParallelDecoder
from .utils import normalize_eos_ids

KNOWN_DECODERS = (
    "threshold",
    "joint_threshold",
    "levenshtein_joint",
    "hierarchy",
)


def load_decoder(config: RunnerConfig | ServerArgs):
    parallel_decoding = getattr(config, "parallel_decoding", "threshold")
    threshold = getattr(config, "threshold", 0.9)
    low_threshold = getattr(config, "low_threshold", 0.3)
    use_credit = getattr(config, "use_credit", False)
    mask_id = getattr(config, "mask_id", 156895)
    eos_id = getattr(config, "eos_id", 156892)
    # The primary eos_id always leads; extra stop tokens (e.g. LLaDA2.2's
    # <|role_end|>) come from RunnerConfig.eos_ids and are deduplicated.
    eos_ids = normalize_eos_ids((eos_id, *(getattr(config, "eos_ids", ()) or ())))

    if parallel_decoding == "threshold":
        if use_credit:
            return CreditThresholdParallelDecoder(
                temperature=0,
                threshold=threshold,
                mask_id=mask_id,
                eos_ids=eos_ids,
            )
        return ThresholdParallelDecoder(
            temperature=0,
            threshold=threshold,
            mask_id=mask_id,
            eos_ids=eos_ids,
        )

    if parallel_decoding == "joint_threshold":
        num_to_transfer = getattr(config, "num_to_transfer", 1)
        if num_to_transfer != 1:
            raise ValueError(
                "joint_threshold only implements num_to_transfer=1 "
                f"(got {num_to_transfer}); the reference two-branch selection "
                "for larger values is not implemented."
            )
        return JointThresholdDecoder(
            temperature=0,
            threshold=threshold,
            editing_threshold=getattr(config, "editing_threshold", 0.5),
            mask_id=mask_id,
            eos_ids=eos_ids,
        )

    if parallel_decoding == "levenshtein_joint":
        block_length = int(getattr(config, "block_length", 32))
        steps = int(getattr(config, "steps", 0) or 0)
        return LevenshteinJointDecoder(
            temperature=0,
            threshold=threshold,
            editing_threshold=getattr(config, "editing_threshold", 0.5),
            mask_id=mask_id,
            eos_ids=eos_ids,
            delete_token_id=getattr(config, "delete_token_id", 156930),
            split_token_id=getattr(config, "split_token_id", 156931),
            steps=steps if steps > 0 else block_length,
            max_post_steps=getattr(config, "max_post_steps", 16),
            max_steps_per_block=getattr(config, "max_steps_per_block", 1000),
        )

    if parallel_decoding == "hierarchy":
        # HierarchyDecoder has no batch_decode and cannot run under the
        # current runners; it is kept only for explicit opt-in use.
        return HierarchyDecoder(
            temperature=0,
            threshold=threshold,
            low_threshold=low_threshold,
            mask_id=mask_id,
            eos_ids=eos_ids,
        )

    raise ValueError(
        f"Unknown parallel_decoding {parallel_decoding!r}; "
        f"expected one of {KNOWN_DECODERS}."
    )
