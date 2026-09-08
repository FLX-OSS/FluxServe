"""Real HTTP benchmark; preserve every response and never count requested tokens.

HumanEval execution/scoring is a separate isolated step. This program only
generates responses and verifies actual decode graph replay when requested.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

from transformers import AutoTokenizer
from test_diffusion_gemma_server import _free_port, _request_json, _wait_ready


def stop_profile_target(pid_file):
    """Signal only our target's process incarnation, even after setproctitle."""
    record = json.loads(pid_file.read_text())
    pid = record["pid"]
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        actual = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        if actual != record["start_time"]:
            return False
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False
    finally:
        os.close(pidfd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["fa4", "flashinfer"], required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--preset", choices=["quality", "speed"], default="quality")
    parser.add_argument("--profile", choices=["nsys", "ncu"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--mini-batch-size", type=int)
    parser.add_argument("--max-scheduled-tokens", type=int, default=2048)
    parser.add_argument("--scheduler-num-device-pages", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.8)
    parser.add_argument("--capture-bs", type=int, nargs="+")
    args = parser.parse_args()
    stop_after_round = False

    def request_stop(_signum, _frame):
        nonlocal stop_after_round
        stop_after_round = True

    # Operators may stop a long allocation cleanly without truncating a round.
    signal.signal(signal.SIGUSR1, request_stop)
    if args.profile and args.repeats != 1:
        parser.error("Profiling is a separate single diagnostic run, not repeated timing")
    # Explicit operator marker for cancelling a queued (not running) command.
    cancel = args.output / "CANCEL"
    if cancel.exists():
        print(f"Cancelled queued benchmark: {cancel.read_text().strip()}", flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    model = "inclusionAI/LLaDA2.1-mini"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    records = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    if args.limit:
        records = records[:args.limit]
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    threshold, editing = ("0.7", "0.5") if args.preset == "quality" else ("0.5", "0.0")
    command = [sys.executable, str(Path(__file__).with_name("llada21_bench_server.py")), "serve", "--model", model,
               "--host", "127.0.0.1", "--port", str(port), "--max-num-seqs", str(args.max_num_seqs or args.concurrency),
               "--mini-batch-size", str(args.mini_batch_size or args.concurrency), "--max-model-len", str(args.max_model_len),
               "--tp-size", "1", "--dp-size", "1", "--ep-size", "1",
               "--max-scheduled-tokens", str(args.max_scheduled_tokens),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--max-new-tokens", "2048", "--attention-backend", args.backend,
               "--kv-cache-layout", "paged", "--scheduler-policy", "paged",
               "--scheduler-num-device-pages", str(args.scheduler_num_device_pages if args.scheduler_num_device_pages is not None else args.concurrency * 64),
               "--block-length", "64", "--page-size", "64", "--apply-template",
               "--parallel-decoding", "joint_threshold", "--threshold", threshold,
               "--editing-threshold", editing, "--max-post-steps", "16"]
    if args.graph:
        command += ["--use-decode-cuda-graph", "--cuda-graph-decode-mode", "padded",
                    "--cuda-graph-capture-bs", *map(str, args.capture_bs or [1, *range(2, (args.max_num_seqs or args.concurrency) + 1, 2)])]
    server_env = os.environ.copy()
    server_pid_file = (args.output / "server.pid").resolve()
    server_env["FLUXSERVE_BENCH_PID_FILE"] = str(server_pid_file)
    if args.profile:
        server_env["FLUXSERVE_BENCH_PROFILE_API"] = args.profile
        artifact = str((args.output / args.profile).resolve())
        if args.profile == "nsys":
            command = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                       "--cuda-graph-trace=node",
                       "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                       "--output", artifact, *command]
        else:
            command = ["ncu", "--target-processes", "all", "--profile-from-start", "off",
                       "--set", "basic", "--kernel-name-base", "demangled",
                       "--kernel-name", "regex:.*(BatchPrefill|flash_attn|FlashAttention).*",
                       "--launch-count", "1", "--export", artifact, *command]
    manifest = dict(vars(args), command=command, model=model, dtype="bfloat16",
                    dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    token_count="retokenized response text (not server token IDs)")
    import torch
    manifest["gpu"] = torch.cuda.get_device_name()
    manifest["hostname"] = os.uname().nodename
    manifest["slurm_job_id"] = os.getenv("SLURM_JOB_ID")
    manifest["packages"] = {}
    for package in ("torch", "transformers", "flash-attn-4", "flashinfer-python"):
        try:
            manifest["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest["packages"][package] = None
    manifest["working_diff_sha256"] = hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest()
    # Include untracked implementation files too; git diff omits them.
    source_paths = [
        Path("python/fluxserve/backend/execution/fa4_cuda_graph_runner.py"),
        Path("python/fluxserve/backend/execution/runners/fa4_diffusion.py"),
        Path("python/fluxserve/backend/execution/runners/flashinfer_diffusion.py"),
        Path("python/fluxserve/backend/execution/flashinfer_cuda_graph_runner.py"),
        Path("python/fluxserve/backend/layers/attention/fa4.py"),
    ]
    manifest["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths if path.exists()
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    def request(item):
        index, sample = item
        payload = {k: sample[k] for k in ("messages", "max_tokens", "temperature") if k in sample}
        payload.update(model=model, stream=False)
        start = time.perf_counter()
        status, body = _request_json(url + "/v1/chat/completions", payload, timeout=1800)
        duration = time.perf_counter() - start
        if status != 200:
            raise RuntimeError(f"request {index}: HTTP {status}: {body}")
        text = body["choices"][0]["message"]["content"]
        return dict(index=index, metadata=sample.get("metadata"), response=body,
                    latency_s=duration, output_tokens=len(tokenizer.encode(text, add_special_tokens=False)))

    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=server_env, start_new_session=True)
        try:
            _wait_ready(url, process, timeout=1200)
            # Warm up the same dataset workload at the measured concurrency.
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                list(pool.map(request, list(enumerate(records))[:args.concurrency]))
            for repeat in range(args.repeats):
                before = _request_json(url + "/metrics")[1]
                started = time.perf_counter()
                results = []
                with (args.output / f"responses_{repeat}.jsonl").open("w") as stream:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                        futures = [pool.submit(request, item) for item in enumerate(records)]
                        for future in concurrent.futures.as_completed(futures):
                            row = future.result()
                            results.append(row)
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                            stream.flush()
                            print(f"{args.backend} graph={args.graph} repeat={repeat} {len(results)}/{len(records)}", flush=True)
                elapsed = time.perf_counter() - started
                after = _request_json(url + "/metrics")[1]
                if args.graph:
                    assert after.get("cuda_graph_decode_replay_count", 0) > before.get("cuda_graph_decode_replay_count", 0), "No actual decode graph replay"
                    assert after.get("cuda_graph_decode_fallback_count", 0) == before.get("cuda_graph_decode_fallback_count", 0), "Unexpected eager decode fallback"
                tokens = sum(row["output_tokens"] for row in results)
                server_tokens = after["generated_tokens"] - before["generated_tokens"]
                latencies = sorted(row["latency_s"] for row in results)
                summary = dict(requests=len(results), elapsed_s=elapsed, output_tokens=tokens,
                               timing_valid_for_comparison=args.profile is None,
                               output_tokens_per_s=tokens / elapsed, requests_per_s=len(results) / elapsed,
                               server_output_tokens=server_tokens,
                               server_output_tokens_per_s=server_tokens / elapsed,
                               latency_p50_s=latencies[len(latencies)//2],
                               latency_mean_s=sum(latencies)/len(latencies),
                               latency_p99_s=latencies[min(len(latencies)-1, int(len(latencies)*.99))],
                               latency_p95_s=latencies[min(len(latencies)-1, int(len(latencies)*.95))],
                               metrics_before=before, metrics_after=after)
                if "benchmark_model_forwards" in after:
                    summary["model_forwards"] = after["benchmark_model_forwards"] - before["benchmark_model_forwards"]
                (args.output / f"summary_{repeat}.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(summary), flush=True)
                if stop_after_round:
                    (args.output / "stopped.json").write_text(json.dumps({
                        "reason": "SIGUSR1: stopped after complete round",
                        "completed_rounds": repeat + 1, "requested_rounds": args.repeats,
                    }, indent=2))
                    break
        finally:
            if args.profile and server_pid_file.exists():
                # Stop the target, not the profiler, so nsys/ncu can finish
                # exporting their reports. A pidfd and birth-time check avoid
                # PID reuse; command lines are unreliable after setproctitle.
                stop_profile_target(server_pid_file)
            elif process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60 if args.profile else 30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    if args.profile:
        suffix = "nsys-rep" if args.profile == "nsys" else "ncu-rep"
        report = args.output / f"{args.profile}.{suffix}"
        if not report.exists() or report.stat().st_size == 0:
            raise RuntimeError(f"Profiler did not export a valid report: {report}")


if __name__ == "__main__":
    main()
