# FA4 TP4/EP4 评测结果（2026-09-15）

## LLaDA2.1-mini

环境：4 × NVIDIA H200，FA4，TP4/EP4，decode CUDA graph。

| 项目 | 结果 |
| --- | --- |
| GSM8K 设置 | 4-shot；1319 道测试题 |
| GSM8K 准确率 | 92.65% |
| Perf 正式请求 | 1000/1000 成功，0 失败 |
| 最大并发 / 目标请求速率 | 16 / 16 req/s |
| 最大输出长度 | 2048 tokens |
| 测试耗时 | 118.5849 s |
| 实际请求吞吐 | 8.4328 req/s |
| 输出 token 吞吐 | 2671.3005 tok/s |
| 总 token 吞吐 | 3939.1765 tok/s |
| 平均请求延迟 | 1.8816 s |
| P95 请求延迟 | 2.93 s |
| 平均输入 / 输出 tokens | 150.351 / 316.776 |

Perf 使用 data/gsm8k.jsonl 的单问题请求，与准确率评测的 4-shot 提示不同。
16 条预热请求不计入正式 perf 结果。非流式输出，不能用工具报告的 TTFT/TPOT/ITL 衡量真实流式延迟。

## LLaDA2.0-flash

正式 perf 未完成：用户报告 GPU 作业到期；日志最后记录进度 208/1000，没有最终 summary。
不将这次运行作为有效性能结果。

## 保留的原始汇总与配置

下方保存准确率报告、完整 perf summary、分位数、压测参数、服务累计指标及 eval 配置。
服务累计指标包含预热；原文件路径和 SHA-256 用于记录来源，原文件已按要求清理。

```json
{
  "sources": {
    "eval_report": {
      "path": "eval-results/gsm8k-4shot-20260915-165902/eval/reports/llada21-fa4-tp4-ep4-4shot/gsm8k.json",
      "sha256": "a06e2a690d330ec54958be3ebabf0dadc716edd6620ae8ed7f18b707cde73347"
    },
    "perf_summary": {
      "path": "perf-results/20260915-182318/llada21-mini/measured/llada21-fa4-graph/parallel_16_number_1000/benchmark_summary.json",
      "sha256": "1c0cf6c64224a21b2e8fc81b936420609eb1a87944169550eea1f8d877895432"
    },
    "perf_percentiles": {
      "path": "perf-results/20260915-182318/llada21-mini/measured/llada21-fa4-graph/parallel_16_number_1000/benchmark_percentile.json",
      "sha256": "f64d341d2878b5ee46b583890c131b1034173b73bcf42644ba0aa99d7b6a2c13"
    },
    "perf_arguments": {
      "path": "perf-results/20260915-182318/llada21-mini/measured/llada21-fa4-graph/parallel_16_number_1000/benchmark_args.json",
      "sha256": "e90df231c791d3df1887a24ffb0b67ac5b71e59a9ac2a3d01cc506574c6622ca"
    },
    "perf_server_metrics": {
      "path": "perf-results/20260915-182318/llada21-mini/server-metrics.json",
      "sha256": "d77990e768822177c8a6d69915a336ef3c028864ed9235b876e99b7a16904983"
    }
  },
  "results": {
    "eval_report": {
      "name": "llada21-fa4-tp4-ep4-4shot@gsm8k",
      "dataset_name": "gsm8k",
      "dataset_pretty_name": "GSM8K",
      "dataset_description": "\n## Overview\n\nGSM8K (Grade School Math 8K) is a high-quality dataset of 8.5K linguistically diverse grade school math word problems created by human problem writers. The dataset is specifically designed to evaluate and improve the multi-step mathematical reasoning capabilities of language models.\n\n## Task Description\n\n- **Task Type**: Mathematical Word Problem Solving\n- **Input**: Natural language math word problem\n- **Output**: Numerical answer derived through step-by-step reasoning\n- **Difficulty**: Grade school level (2-8 reasoning steps required)\n\n## Key Features\n\n- Problems require basic arithmetic operations (addition, subtraction, multiplication, division)\n- Solutions involve 2 to 8 sequential reasoning steps\n- High linguistic diversity in problem formulations\n- Human-written problems ensuring natural language quality\n- Clear numerical answers for objective evaluation\n\n## Evaluation Notes\n\n- Default configuration uses **4-shot** examples with Chain-of-Thought (CoT) prompting\n- Answers should be formatted within `\\boxed{}` for proper extraction\n- The metric extracts numerical values for accuracy comparison\n- Supports both zero-shot and few-shot evaluation modes\n",
      "model_name": "llada21-fa4-tp4-ep4-4shot",
      "score": 0.9265,
      "metrics": [
        {
          "name": "mean_acc",
          "num": 1319,
          "score": 0.9265,
          "macro_score": 0.9265,
          "categories": [
            {
              "name": [
                "default"
              ],
              "num": 1319,
              "score": 0.9265,
              "macro_score": 0.9265,
              "subsets": [
                {
                  "name": "main",
                  "score": 0.9265,
                  "num": 1319
                }
              ]
            }
          ]
        }
      ],
      "analysis": "N/A",
      "perf_metrics": null,
      "num": 1319
    },
    "perf_summary": {
      "Test Duration (s)": 118.5849,
      "Concurrency": 16,
      "Request Rate (req/s)": 16.0,
      "Total Requests": 1000,
      "Success Requests": 1000,
      "Failed Requests": 0,
      "Req Throughput (req/s)": 8.4328,
      "Avg Latency (s)": 1.8816,
      "Avg Input Tokens": 150.351,
      "Output Throughput (tok/s)": 2671.3005,
      "Total Throughput (tok/s)": 3939.1765,
      "TTFT (ms)": 1881.61,
      "TPOT (ms)": 0.0,
      "ITL (ms)": 0.0,
      "Avg Output Tokens": 316.776,
      "Input Throughput (tok/s)": 0.0
    },
    "perf_percentiles": [
      {
        "Percentiles": "1%",
        "Latency (s)": 0.73,
        "TTFT (ms)": 732.57,
        "ITL (ms)": null,
        "TPOT (ms)": 0.0,
        "Input tokens": 116.0,
        "Output tokens": 114.0,
        "Output (tok/s)": 95.13,
        "Total (tok/s)": 162.79,
        "Decode (tok/s)": null
      },
      {
        "Percentiles": "5%",
        "Latency (s)": 0.96,
        "TTFT (ms)": 956.46,
        "ITL (ms)": null,
        "TPOT (ms)": 0.0,
        "Input tokens": 123.0,
        "Output tokens": 138.0,
        "Output (tok/s)": 110.85,
        "Total (tok/s)": 193.24,
        "Decode (tok/s)": null
      },
      {
        "Percentiles": "10%",
        "Latency (s)": 1.07,
        "TTFT (ms)": 1071.4,
        "ITL (ms)": null,
        "TPOT (ms)": 0.0,
        "Input tokens": 126.0,
        "Output tokens": 161.0,
        "Output (tok/s)": 123.59,
        "Total (tok/s)": 205.68,
        "Decode (tok/s)": null
      },
      {
        "Percentiles": "25%",
        "Latency (s)": 1.31,
        "TTFT (ms)": 1307.06,
        "ITL (ms)": null,
        "TPOT (ms)": 0.0,
        "Input tokens": 134.0,
        "Output tokens": 210.0,
        "Output (tok/s)": 142.68,
        "Total (tok/s)": 224.61,
        "Decode (tok/s)": null
      },
      {
        "Percentiles": "50%",
        "Latency (s)": 1.66,
        "TTFT (ms)": 1661.49,
        "ITL (ms)": null,
        "TPOT (ms)": 0.0,
        "Input tokens": 147.0,
        "Output tokens": 268.0,
        "Output (tok/s)": 165.61,
        "Total (tok/s)": 254.27,
        "Decode (tok/s)": null
      },
      {
        "Percentiles": "75%",
        "Latency (s)": 2.03,
        "TTFT (ms)": 2030.68,
        "ITL (ms)": null,
        "TPOT (ms)": 0.0,
        "Input tokens": 163.0,
        "Output tokens": 340.0,
        "Output (tok/s)": 186.8,
        "Total (tok/s)": 289.21,
        "Decode (tok/s)": null
      },
      {
        "Percentiles": "90%",
        "Latency (s)": 2.51,
        "TTFT (ms)": 2507.04,
        "ITL (ms)": null,
        "TPOT (ms)": 0.0,
        "Input tokens": 180.0,
        "Output tokens": 435.0,
        "Output (tok/s)": 209.84,
        "Total (tok/s)": 321.46,
        "Decode (tok/s)": null
      },
      {
        "Percentiles": "95%",
        "Latency (s)": 2.93,
        "TTFT (ms)": 2929.55,
        "ITL (ms)": null,
        "TPOT (ms)": 0.0,
        "Input tokens": 192.0,
        "Output tokens": 514.0,
        "Output (tok/s)": 221.65,
        "Total (tok/s)": 342.7,
        "Decode (tok/s)": null
      },
      {
        "Percentiles": "99%",
        "Latency (s)": 9.59,
        "TTFT (ms)": 9587.56,
        "ITL (ms)": null,
        "TPOT (ms)": 0.0,
        "Input tokens": 214.0,
        "Output tokens": 2048.0,
        "Output (tok/s)": 249.62,
        "Total (tok/s)": 408.09,
        "Decode (tok/s)": null
      }
    ],
    "perf_arguments": {
      "model": "inclusionAI/LLaDA2.1-mini",
      "model_id": "LLaDA2.1-mini",
      "attn_implementation": null,
      "api": "openai",
      "tokenizer_path": "inclusionAI/LLaDA2.1-mini",
      "port": 8877,
      "url": "http://127.0.0.1:8000/v1/chat/completions",
      "headers": {},
      "connect_timeout": null,
      "read_timeout": null,
      "total_timeout": 21600,
      "api_key": null,
      "no_test_connection": false,
      "number": 1000,
      "parallel": 16,
      "rate": 16.0,
      "open_loop": false,
      "sleep_interval": 5,
      "sla_auto_tune": false,
      "sla_variable": "parallel",
      "sla_params": null,
      "sla_num_runs": 3,
      "sla_upper_bound": 65536,
      "sla_lower_bound": 1,
      "sla_fixed_parallel": null,
      "sla_number_multiplier": null,
      "db_commit_interval": 1000,
      "queue_size_multiplier": 5,
      "in_flight_task_multiplier": 2,
      "log_every_n_query": 100,
      "debug": false,
      "enable_progress_tracker": false,
      "visualizer": null,
      "wandb_api_key": null,
      "swanlab_api_key": null,
      "name": "llada21-fa4-graph",
      "outputs_dir": "/u/dzhu8/workspace/FluxServe/perf-results/20260915-182318/llada21-mini/measured/llada21-fa4-graph/parallel_16_number_1000",
      "no_timestamp": true,
      "max_prompt_length": 131072,
      "min_prompt_length": 0,
      "prefix_length": 0,
      "prompt": null,
      "query_template": null,
      "apply_chat_template": true,
      "image_width": 224,
      "image_height": 224,
      "image_format": "RGB",
      "image_num": 1,
      "image_patch_size": 28,
      "dataset": "line_by_line",
      "dataset_path": "/u/dzhu8/workspace/FluxServe/data/gsm8k.jsonl",
      "dataset_offset": 0,
      "frequency_penalty": null,
      "repetition_penalty": null,
      "logprobs": null,
      "max_tokens": 2048,
      "min_tokens": null,
      "n_choices": null,
      "seed": null,
      "stop": null,
      "stop_token_ids": null,
      "stream": false,
      "temperature": 0.0,
      "top_p": null,
      "top_k": null,
      "extra_args": {},
      "tokenize_prompt": false,
      "multi_turn": false,
      "min_turns": 1,
      "max_turns": null,
      "multi_turn_args": null
    },
    "perf_server_metrics": {
      "total_requests": 1018,
      "successful_requests": 1018,
      "failed_requests": 0,
      "aborted_requests": 0,
      "waiting_requests": 0,
      "running_requests": 0,
      "prompt_tokens": 144675,
      "generated_tokens": 322563,
      "queue_latency_avg_s": 0.14637899469064589,
      "execution_latency_avg_s": 1.7197970461049108,
      "e2e_latency_avg_s": 1.8661849503432837,
      "cuda_graph_decode_capture_count": 7,
      "cuda_graph_decode_replay_count": 14243,
      "cuda_graph_decode_fallback_count": 0,
      "cuda_graph_decode_padded_rows": 9555,
      "cuda_graph_capture_time_s": 9.335134024266154,
      "cuda_graph_capture_memory_bytes": 2404677120
    }
  },
  "eval_task_config_yaml": "analysis_report: false\napi_url: http://127.0.0.1:8000/v1/chat/completions\nchat_template: null\ncollect_perf: false\ndataset_args:\n  gsm8k:\n    aggregation: mean\n    data_statistics: null\n    dataset_id: /u/dzhu8/workspace/FluxServe/data/gsm8k-eval\n    default_subset: default\n    description: '\n\n      ## Overview\n\n\n      GSM8K (Grade School Math 8K) is a high-quality dataset of 8.5K linguistically\n      diverse grade school math word problems created by human problem writers. The\n      dataset is specifically designed to evaluate and improve the multi-step mathematical\n      reasoning capabilities of language models.\n\n\n      ## Task Description\n\n\n      - **Task Type**: Mathematical Word Problem Solving\n\n      - **Input**: Natural language math word problem\n\n      - **Output**: Numerical answer derived through step-by-step reasoning\n\n      - **Difficulty**: Grade school level (2-8 reasoning steps required)\n\n\n      ## Key Features\n\n\n      - Problems require basic arithmetic operations (addition, subtraction, multiplication,\n      division)\n\n      - Solutions involve 2 to 8 sequential reasoning steps\n\n      - High linguistic diversity in problem formulations\n\n      - Human-written problems ensuring natural language quality\n\n      - Clear numerical answers for objective evaluation\n\n\n      ## Evaluation Notes\n\n\n      - Default configuration uses **4-shot** examples with Chain-of-Thought (CoT)\n      prompting\n\n      - Answers should be formatted within `\\boxed{}` for proper extraction\n\n      - The metric extracts numerical values for accuracy comparison\n\n      - Supports both zero-shot and few-shot evaluation modes\n\n      '\n    eval_split: test\n    extra_params: {}\n    few_shot_num: 4\n    few_shot_prompt_template: 'Here are some examples of how to solve similar problems:\n\n\n      {fewshot}\n\n\n      {question}\n\n      Please reason step by step, and put your final answer within \\boxed{{}}.'\n    few_shot_random: false\n    filters: null\n    force_redownload: false\n    metric_list:\n    - acc:\n        numeric: true\n    name: gsm8k\n    output_types:\n    - generation\n    paper_url: https://arxiv.org/abs/2110.14168\n    pretty_name: GSM8K\n    prompt_template: '{question}\n\n      Please reason step by step, and put your final answer within \\boxed{{}}.'\n    query_template: null\n    review_timeout: null\n    sample_example: null\n    sandbox_config: {}\n    shuffle: false\n    shuffle_choices: false\n    subset_list:\n    - main\n    system_prompt: null\n    tags:\n    - Math\n    - Reasoning\n    train_split: train\ndataset_dir: /u/dzhu8/.cache/modelscope/hub/datasets\ndataset_hub: modelscope\ndatasets:\n- gsm8k\ndebug: false\nenable_progress_tracker: false\neval_backend: Native\neval_batch_size: 4\neval_config: null\neval_type: openai_api\nevalscope_version: 0.0.0_dev\ngeneration_config:\n  batch_size: 4\n  max_tokens: 2048\n  temperature: 0.0\nignore_errors: false\njudge_model_args: {}\njudge_strategy: rule\njudge_worker_num: 1\nlimit: 1319\nmodel: inclusionAI/LLaDA2.1-mini\nmodel_args: {}\nmodel_id: llada21-fa4-tp4-ep4-4shot\nmodel_task: text_generation\nno_timestamp: true\nrepeats: 1\nrerun_review: false\nsandbox_manager_config: {}\nsandbox_type: docker\nseed: 42\nstream: null\ntimeout: null\nuse_cache: null\nuse_sandbox: false\nwork_dir: /u/dzhu8/workspace/FluxServe/eval-results/gsm8k-4shot-20260915-165902/eval\n",
  "llada20_flash": {
    "status": "incomplete",
    "reason": "GPU allocation expired (reported by user)",
    "last_logged_progress": 208,
    "planned_requests": 1000,
    "valid_final_perf_result": false
  },
  "notes": [
    "Warmup excluded from measured performance.",
    "Non-streaming test: reported TTFT/TPOT/ITL are not valid streaming latency measurements.",
    "Non-finite percentile values normalized to null."
  ]
}
```
