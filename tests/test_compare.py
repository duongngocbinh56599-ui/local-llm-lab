from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_llm_lab.compare import CompareRequest, compare_plans, parse_csv, parse_int_csv, write_compare_outputs


class CompareTest(unittest.TestCase):
    def test_csv_parsing(self) -> None:
        self.assertEqual(parse_csv("Q6_K,Q4_K_M", ["Q4_K_M"]), ["Q6_K", "Q4_K_M"])
        self.assertEqual(parse_int_csv("4096,8192", [1024]), [4096, 8192])
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_int_csv("4096,0", [1024])

    def test_compare_finds_best_candidate(self) -> None:
        data = compare_plans(
            CompareRequest(
                model_name="llama-3.3-70b",
                params=None,
                quantizations=["Q6_K", "Q4_K_M", "Q3_K_M"],
                contexts=[4096, 8192, 32768],
                backends=["auto"],
                concurrency=1,
                model_format=None,
                hardware_fixture="apple-m4-max-128gb",
            )
        )
        summary = data["compare"]["summary"]
        self.assertEqual(summary["total_plans"], 9)
        self.assertGreater(summary["runnable_plans"], 0)
        self.assertIn(summary["best"]["verdict"], {"smooth", "tight"})

    def test_compare_outputs(self) -> None:
        data = compare_plans(
            CompareRequest(
                model_name=None,
                params="600B",
                quantizations=["Q4_K_M", "IQ2_XS"],
                contexts=[8192, 32768],
                backends=["auto"],
                concurrency=1,
                model_format=None,
                hardware_fixture="apple-m4-max-128gb",
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = write_compare_outputs(data, tmp)
            names = {Path(item).name for item in result["files"]}
            self.assertIn("compare.json", names)
            self.assertIn("compare.md", names)
            self.assertIn("index.html", names)
            self.assertIn("compare_decode.svg", names)
            html = (Path(tmp) / "index.html").read_text(encoding="utf-8")
            self.assertIn("verdictFilter", html)
            self.assertIn("Copy best deploy dry-run", html)
            self.assertIn("Copy recommend command", html)
            self.assertIn("Recommendation reason", html)
            self.assertIn("data-sort=\"margin\"", html)
            self.assertIn("python3 -m local_llm_lab deploy", html)
            self.assertIn("python3 -m local_llm_lab recommend", html)


if __name__ == "__main__":
    unittest.main()
