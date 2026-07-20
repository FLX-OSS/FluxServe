import argparse
import unittest

from fluxserve.bench import RequestOutput, _parse_metrics, summarize


class ParseMetricsTest(unittest.TestCase):
    def test_normalizes_deduplicates_and_orders_metrics(self):
        self.assertEqual(
            _parse_metrics(" http_overhead, e2e,QUEUE,queue "),
            ("E2E", "QUEUE", "HTTP_OVERHEAD"),
        )

    def test_rejects_unknown_metrics(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_metrics("E2E,TTFT")

    def test_rejects_empty_metrics(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_metrics(" , ")

    def test_requires_e2e(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_metrics("QUEUE")


class SummarizeMetricsTest(unittest.TestCase):
    def setUp(self):
        self.outputs = [
            RequestOutput(
                success=True,
                client_latency=0.15,
                server_e2e_latency=0.10,
                queue_latency=0.02,
                execution_latency=0.08,
            )
        ]

    def test_default_output_contains_only_e2e_statistics(self):
        result = summarize(self.outputs, 1.0, [50.0])

        self.assertEqual(result["metrics"], ["E2E"])
        self.assertEqual(result["mean_e2e_ms"], 100.0)
        self.assertIn("e2els", result)
        self.assertNotIn("mean_server_e2e_ms", result)
        self.assertNotIn("server_e2els", result)
        self.assertNotIn("mean_queue_ms", result)
        self.assertNotIn("queue_latencies", result)
        self.assertNotIn("execution_latencies", result)
        self.assertNotIn("http_overheads", result)

    def test_selected_metrics_add_statistics_and_details(self):
        metrics = ("E2E", "QUEUE", "EXECUTION", "HTTP_OVERHEAD")
        result = summarize(self.outputs, 1.0, [50.0], metrics)

        self.assertEqual(result["metrics"], list(metrics))
        self.assertEqual(result["mean_queue_ms"], 20.0)
        self.assertEqual(result["mean_execution_ms"], 80.0)
        self.assertAlmostEqual(result["mean_http_overhead_ms"], 50.0)
        self.assertEqual(result["queue_latencies"], [0.02])
        self.assertEqual(result["execution_latencies"], [0.08])
        self.assertAlmostEqual(result["http_overheads"][0], 0.05)

    def test_e2e_uses_client_fallback(self):
        output = RequestOutput(success=True, client_latency=0.12)
        result = summarize([output], 1.0, [50.0], ("E2E", "HTTP_OVERHEAD"))

        self.assertEqual(result["mean_e2e_ms"], 120.0)
        self.assertEqual(result["mean_http_overhead_ms"], 0.0)
        self.assertEqual(result["e2e_sources"], ["client_fallback"])


if __name__ == "__main__":
    unittest.main()
