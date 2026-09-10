from __future__ import annotations

import unittest

from local_llm_lab.models import apply_overrides, generic_model_from_params, get_model
from local_llm_lab.quantization import get_quantization
from local_llm_lab.units import parse_gib, parse_params


class UnitsAndModelsTest(unittest.TestCase):
    def test_parse_params(self) -> None:
        self.assertEqual(parse_params("600B"), 600)
        self.assertEqual(parse_params("120b"), 120)
        self.assertAlmostEqual(parse_params("7000M"), 7)
        self.assertAlmostEqual(parse_params(70_000_000_000), 70)

    def test_parse_params_rejects_nonpositive_or_nonfinite_values(self) -> None:
        for value in ("0B", 0, 0.0, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "greater than 0"):
                    parse_params(value)

    def test_parse_gib(self) -> None:
        self.assertEqual(parse_gib("128GB"), 128)
        self.assertEqual(parse_gib("1TB"), 1024)
        self.assertAlmostEqual(parse_gib("512MB"), 0.5)

    def test_model_preset_lookup(self) -> None:
        model = get_model("llama-3.3-70b")
        self.assertEqual(model.params_b, 70)
        self.assertEqual(model.kv_heads, 8)

    def test_generic_model_for_600b(self) -> None:
        model = generic_model_from_params("600B")
        self.assertEqual(model.params_b, 600)
        self.assertEqual(model.confidence, "low")
        self.assertGreaterEqual(model.layers, 100)

    def test_overrides_take_precedence(self) -> None:
        model = apply_overrides(get_model("llama-3.3-70b"), params="120B", layers=96, kv_heads=16)
        self.assertEqual(model.params_b, 120)
        self.assertEqual(model.layers, 96)
        self.assertEqual(model.kv_heads, 16)
        self.assertEqual(model.confidence, "low")

    def test_quant_alias(self) -> None:
        self.assertEqual(get_quantization("q4").name, "Q4_K_M")
        self.assertLess(get_quantization("IQ2_XS").bytes_per_param, get_quantization("Q4_K_M").bytes_per_param)


if __name__ == "__main__":
    unittest.main()
