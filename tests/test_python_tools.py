#!/usr/bin/env python3
"""
Unit tests for Python custom tool contracts (VT-07 / VS-07).
Validates:
  - mean.py missing input
  - mean.py empty batch
  - mean.py read-only views with ragged/different shapes
  - mean.py float64 dense rank=1 output
  - counter.py lifecycle and counting
"""

import importlib.util
import pathlib
import unittest
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

def load_module_from_file(file_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("custom_tool", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class TestPythonToolContracts(unittest.TestCase):
    def setUp(self):
        self.mean_py = ROOT / "assets" / "tools" / "custom" / "Python" / "示例" / "mean.py"
        self.counter_py = ROOT / "assets" / "tools" / "custom" / "Python" / "示例" / "counter.py"
        self.assertTrue(self.mean_py.exists(), f"mean.py not found at {self.mean_py}")
        self.assertTrue(self.counter_py.exists(), f"counter.py not found at {self.counter_py}")

    def test_mean_missing_input(self):
        mod = load_module_from_file(self.mean_py)
        instance = mod.ToolInstance()
        init_res = instance.initialize({})
        self.assertEqual(init_res.get("__status_code"), 0)

        res = instance.execute({})
        self.assertNotEqual(res.get("__status_code"), 0)
        self.assertIsNone(res.get("means"))

        rel_res = instance.release()
        self.assertEqual(rel_res.get("__status_code"), 0)

    def test_mean_empty_batch(self):
        mod = load_module_from_file(self.mean_py)
        instance = mod.ToolInstance()
        instance.initialize({})

        res = instance.execute({"images": []})
        self.assertEqual(res.get("__status_code"), 0)
        means = res.get("means")
        self.assertIsInstance(means, np.ndarray)
        self.assertEqual(means.dtype, np.float64)
        self.assertEqual(means.shape, (0,))
        self.assertEqual(means.ndim, 1)
        self.assertTrue(means.flags.c_contiguous)

    def test_mean_ragged_readonly_inputs(self):
        mod = load_module_from_file(self.mean_py)
        instance = mod.ToolInstance()
        instance.initialize({})

        img1 = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        img1.setflags(write=False) # Read-only view
        img2 = np.array([[6.0, 7.0], [7.0, 8.0]], dtype=np.uint8)
        img2.setflags(write=False) # Read-only view

        res = instance.execute({"images": [img1, img2]})
        self.assertEqual(res.get("__status_code"), 0)
        means = res.get("means")
        self.assertIsInstance(means, np.ndarray)
        self.assertEqual(means.dtype, np.float64)
        self.assertEqual(means.shape, (2,))
        self.assertEqual(means.ndim, 1)
        self.assertTrue(means.flags.c_contiguous)
        self.assertAlmostEqual(means[0], 2.5, places=5)
        self.assertAlmostEqual(means[1], 7.0, places=5)

    def test_counter_tool(self):
        mod = load_module_from_file(self.counter_py)
        instance = mod.CounterTool()
        init_res = instance.initialize({})
        self.assertEqual(init_res.get("__status_code"), 0)

        r1 = instance.execute({})
        self.assertEqual(r1.get("__status_code"), 0)
        self.assertEqual(r1.get("count"), 1)

        r2 = instance.execute({})
        self.assertEqual(r2.get("__status_code"), 0)
        self.assertEqual(r2.get("count"), 2)

        rel = instance.release()
        self.assertEqual(rel.get("__status_code"), 0)

if __name__ == "__main__":
    unittest.main()
