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

import math
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

def add_gumbel_noise(logits, temperature):
    '''
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
    '''
    if math.isclose(temperature, 0.0):
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def broadcast_if_needed(x, src=0, group=None):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.broadcast(x, src=src)


@torch.no_grad()
@torch.compile(dynamic=True)
def get_transfer_index_hierarchy_fast_v2(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    mask_id,
    threshold=None,
    low_threshold=None,
):
    if not math.isclose(temperature, 0.0):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits

    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float32), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if num_transfer_tokens is not None:
        assert threshold is None
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(
                confidence[j], k=num_transfer_tokens[j]
            )
            transfer_index[j, select_index] = True

    else:
        for i in range(mask_index.shape[0]):

            mask_i = mask_index[i].int()
            conf_i = confidence[i]

            if low_threshold is not None:
                max_value, max_index = torch.max(conf_i, dim=0)
                if max_value < low_threshold:
                    transfer_index[i, max_index] = True
                    continue

            diff = torch.diff(
                torch.cat([mask_i[:1] * 0, mask_i, mask_i[-1:] * 0])
            )
            starts = (diff == 1).nonzero(as_tuple=True)[0]
            ends = (diff == -1).nonzero(as_tuple=True)[0]

            if len(starts) > 0:
                max_indices = [
                    s + torch.argmax(conf_i[s:e])
                    for s, e in zip(starts.tolist(), ends.tolist())
                ]
                transfer_index[i, max_indices] = True

            if low_threshold is not None:
                transfer_index[i] = torch.logical_and(
                    transfer_index[i], conf_i > low_threshold
                )

        if threshold is not None:
            transfer_index = torch.logical_or(
                transfer_index, confidence > threshold
            )

    return x0, transfer_index


@torch.no_grad()
def get_transfer_index_hierarchy_remask(
    logits,
    temperature,
    mask_index,
    x,
    num_transfer_tokens,
    mask_id,
    threshold=None,
    low_threshold=None,
    remask_threshold=0.4,
):
    if not math.isclose(temperature, 0.0):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits

    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
    )  # b, l

    lower_index = x0_p < remask_threshold
    remask_index = torch.logical_and(
        lower_index, torch.logical_not(mask_index)
    )
    mask_new = torch.logical_or(lower_index, mask_index)

    confidence = torch.where(mask_new, x0_p, float('-inf'))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

    remask_cnt = remask_index.sum(dim=1)

    if num_transfer_tokens is not None:
        assert threshold is None
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(
                confidence[j], k=num_transfer_tokens[j]
            )
            transfer_index[j, select_index] = True

    else:
        for i in range(mask_new.shape[0]):
            mask_i = mask_new[i].int()
            conf_i = confidence[i]

            diff = torch.diff(
                torch.cat([mask_i[:1] * 0, mask_i, mask_i[-1:] * 0])
            )
            starts = (diff == 1).nonzero(as_tuple=True)[0]
            ends = (diff == -1).nonzero(as_tuple=True)[0]

            if len(starts) > 0:
                max_indices = [
                    s + torch.argmax(conf_i[s:e])
                    for s, e in zip(starts.tolist(), ends.tolist())
                ]
                transfer_index[i, max_indices] = True

            if low_threshold is not None:
                transfer_index[i] = torch.logical_and(
                    transfer_index[i], conf_i > low_threshold
                )

            if threshold is not None:
                transfer_index[i] = torch.logical_or(
                    transfer_index[i], conf_i > threshold
                )

            gap = int((remask_cnt[i] + 1 - transfer_index[i].sum()).item())
            if gap > 0:
                conf_i[transfer_index[i]] = float('-inf')
                values, indices = torch.topk(
                    conf_i, gap, largest=True, sorted=False
                )
                transfer_index[i][indices] = True

    remask_index = torch.logical_and(
        remask_index, torch.logical_not(transfer_index)
    )
    x0[remask_index] = mask_id
    transfer_index[remask_index] = True

    return x0, transfer_index


def get_transfer_index_cache(
    logits,
    mask_index,
    x,
    block_end,
    num_transfer_tokens,
    temperature,
    remasking,
    threshold=None,
    minimal_topk=1,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits[mask_index].to(torch.float32), dim=-1).to(
            logits.dtype
        )
        x0_p = torch.squeeze(
            torch.gather(
                p,
                dim=-1,
                index=torch.unsqueeze(x0[mask_index], -1),
            ),
            -1,
        )  # b, l
        confidence = torch.full(
            x0.shape, -np.inf, device=x0.device, dtype=logits.dtype
        )
        confidence[mask_index] = x0_p
        confidence[:, block_end:] = -np.inf

    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        x0_p[:, block_end:] = -np.inf
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)
    else:
        raise NotImplementedError(remasking)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(minimal_topk, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    threshold=None,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index_dynamic(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    factor=1,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        ns = list(range(1, num_transfer_tokens[j] + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]

        # at least one token is transferred
        threshs[0] = -1
        sorted_confidence = torch.sort(
            confidence[j][mask_index[j]], dim=-1, descending=True
        )[0]
        assert len(sorted_confidence) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index
