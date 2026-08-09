from flux_scheduler import RequestSpec, Scheduler, SchedulerConfig


def test_scheduler_prefills_only_aligned_prompt_prefix():
    cfg = SchedulerConfig()
    cfg.page_size = 64
    cfg.decode_input_tokens = 64
    cfg.max_batch_size = 1
    cfg.max_scheduled_tokens = 512
    cfg.num_device_pages = 16
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True
    scheduler = Scheduler(cfg)

    request = RequestSpec()
    request.request_id = "prompt-100"
    request.tokens = list(range(100))
    request.prefill_length = 64
    scheduler.submit_requests([request])

    plan = scheduler.next_execution_plan()
    op = list(plan.forward)[0]
    assert list(op.input_lengths) == [64]
    assert list(op.prefill_lengths) == [64]
    assert list(op.input_ids) == list(range(64))

