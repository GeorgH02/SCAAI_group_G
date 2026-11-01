import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench import quantization


FIXTURE = {"scenario": "B", "question": "Average superhost price in Leopoldstadt?",
           "sql_evidence": [{"tool": "compare_superhost_prices", "groups": []}],
           "review_evidence": [{"review_id": 1, "listing_id": 2, "text": "Great host"}], "landmarks": []}


class QuantizationBenchTests(unittest.TestCase):
    def test_each_run_gets_a_unique_prompt_start(self):
        first = quantization.ollama_messages(FIXTURE, "aaaa")
        second = quantization.ollama_messages(FIXTURE, "bbbb")
        self.assertNotEqual(first[0]["content"][:30], second[0]["content"][:30])
        self.assertEqual(first[1], second[1])
        self.assertIn("Average superhost price", first[1]["content"])

    def test_report_summarises_each_precision(self):
        rows = []
        for model, quant, rate in (("qwen2.5:7b-instruct-fp16", "FP16", 4.0), ("qwen2.5:7b", "INT4 (Q4_K_M)", 16.0)):
            for repeat in range(2):
                rows.append({"label": "test", "model": model, "quant": quant, "scenario": "B", "repeat": repeat,
                             "load_s": 5, "size_gib": 4.4, "gpu_share": 1.0, "json_valid": True, "answer": "Ok.",
                             "ttft_s": 10.0, "total_s": 20.0, "prompt_tokens": 2000, "prompt_tok_s": 200.0,
                             "output_tokens": 100, "gen_tok_s": rate,
                             "guard": {"sentences": 2, "supported": 1, "removed": 1, "verified": True}})
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "quantization_test.jsonl"
            results.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with mock.patch.object(quantization, "REPORT", Path(tmp) / "report.md"), \
                    mock.patch.object(quantization, "FINDINGS", Path(tmp) / "none.md"), \
                    mock.patch("builtins.print"):
                quantization.report([results])
            text = (Path(tmp) / "report.md").read_text()
        self.assertLess(text.index("| FP16 |"), text.index("| INT4 (Q4_K_M) |"))
        self.assertIn("| 4.0 |", text)
        self.assertIn("| 2/2 | 0/2 | 2/4 (in 2 valid answers) |", text)
        self.assertIn("## test", text)


if __name__ == "__main__":
    unittest.main()
