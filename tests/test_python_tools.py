#!/usr/bin/env python3
"""Public lifecycle and input/output contracts for Python example tools."""

import importlib.util
import json
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
        mean = json.loads((ROOT / "assets/tools/custom/Python/示例/计算图像均值/计算图像均值.json").read_text(encoding="utf-8"))["entry"]
        counter = json.loads((ROOT / "assets/tools/custom/Python/示例/调用计数器/调用计数器.json").read_text(encoding="utf-8"))["entry"]
        roi = json.loads((ROOT / "assets/tools/custom/Python/示例/ROI几何/ROI几何.json").read_text(encoding="utf-8"))["entry"]
        self.mean_py = ROOT / "python/custom" / mean["package"] / mean["file"]
        self.counter_py = ROOT / "python/custom" / counter["package"] / counter["file"]
        self.roi_py = ROOT / "python/custom" / roi["package"] / roi["file"]
        self.assertTrue(self.mean_py.exists(), f"mean.py not found at {self.mean_py}")
        self.assertTrue(self.counter_py.exists(), f"counter.py not found at {self.counter_py}")
        self.assertTrue(self.roi_py.exists(), f"ROI tool.py not found at {self.roi_py}")

    def test_mean_missing_input(self):
        mod = load_module_from_file(self.mean_py)
        instance = mod.ToolInstance()
        init_res = instance.initialize({})
        self.assertEqual(init_res.get("__status_code"), 0)

        for inputs in ({}, {"images": None}):
            with self.subTest(inputs=inputs):
                res = instance.execute(inputs)
                self.assertEqual(res.get("__status_code"), 1)
                self.assertIn("输入端口 'images' 缺失", res.get("__status_message"))
                self.assertIsNone(res.get("means"))

        rel_res = instance.release()
        self.assertEqual(rel_res.get("__status_code"), 0)

    def test_mean_rejects_non_batch_inputs(self):
        instance = load_module_from_file(self.mean_py).ToolInstance()
        instance.initialize({})
        image = np.ones((2, 3), dtype=np.uint8)
        for images in (image, (image,), {"image": image}, 7):
            with self.subTest(container=type(images).__name__):
                result = instance.execute({"images": images})
                self.assertEqual(result["__status_code"], 1)
                self.assertIn("输入端口 'images'", result["__status_message"])
                self.assertIn("list", result["__status_message"])
                self.assertIsNone(result["means"])

    def test_mean_rejects_non_image_elements(self):
        instance = load_module_from_file(self.mean_py).ToolInstance()
        instance.initialize({})
        image = np.ones((2, 3), dtype=np.uint8)
        for invalid in (np.float32(2), "bad", [[1, 2]], None):
            with self.subTest(element=type(invalid).__name__):
                result = instance.execute({"images": [image, invalid]})
                self.assertEqual(result["__status_code"], 1)
                self.assertIn("输入端口 'images'", result["__status_message"])
                self.assertIn("批次索引 1", result["__status_message"])
                self.assertIn("ndarray", result["__status_message"])
                self.assertIsNone(result["means"])

    def test_mean_rejects_invalid_image_shapes(self):
        instance = load_module_from_file(self.mean_py).ToolInstance()
        instance.initialize({})
        valid = np.ones((2, 3), dtype=np.uint8)
        for shape in ((), (3,), (1, 2, 3, 4), (0, 3), (2, 0), (2, 3, 0)):
            with self.subTest(shape=shape):
                result = instance.execute({"images": [valid, np.zeros(shape, dtype=np.uint8)]})
                self.assertEqual(result["__status_code"], 1)
                self.assertIn("输入端口 'images'", result["__status_message"])
                self.assertIn("批次索引 1", result["__status_message"])
                self.assertIn("图像形状", result["__status_message"])
                self.assertIn(str(shape), result["__status_message"])
                self.assertIsNone(result["means"])

    def test_mean_rejects_non_real_pixels(self):
        instance = load_module_from_file(self.mean_py).ToolInstance()
        instance.initialize({})
        image = np.ones((2, 3), dtype=np.uint8)
        for dtype in (np.str_, object, np.bool_, np.complex64):
            with self.subTest(dtype=dtype):
                invalid = np.full((2, 3), 2, dtype=dtype)
                result = instance.execute({"images": [image, invalid]})
                self.assertEqual(result["__status_code"], 1)
                self.assertIn("输入端口 'images'", result["__status_message"])
                self.assertIn("批次索引 1", result["__status_message"])
                self.assertIn("实数数值", result["__status_message"])
                self.assertIn(str(invalid.dtype), result["__status_message"])
                self.assertIsNone(result["means"])

    def test_mean_empty_batch(self):
        mod = load_module_from_file(self.mean_py)
        instance = mod.ToolInstance()
        instance.initialize({})

        res = instance.execute({"images": []})
        self.assertEqual(res.get("__status_code"), 0)
        self.assertEqual(res.get("__status_message"), "运行成功")
        means = res.get("means")
        self.assertIsInstance(means, list)
        self.assertEqual(means, [])

    def test_mean_ragged_inputs_not_modified(self):
        mod = load_module_from_file(self.mean_py)
        instance = mod.ToolInstance()
        instance.initialize({})

        img1 = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        img1.setflags(write=False) # Read-only view
        img2 = np.array([[6.0, 7.0], [7.0, 8.0]], dtype=np.uint8)
        img2.setflags(write=False) # Read-only view

        res = instance.execute({"images": [img1, img2]})
        self.assertEqual(res.get("__status_code"), 0)
        self.assertEqual(res.get("__status_message"), "运行成功")
        means = res.get("means")
        self.assertIsInstance(means, list)
        self.assertEqual(len(means), 2)
        self.assertTrue(all(isinstance(value, float) for value in means))
        self.assertAlmostEqual(means[0], 2.5, places=5)
        self.assertAlmostEqual(means[1], 7.0, places=5)

    def test_mean_real_pixel_types_and_strided_color_inputs(self):
        instance = load_module_from_file(self.mean_py).ToolInstance()
        instance.initialize({})
        for dtype in (np.uint8, np.int8, np.uint16, np.int16,
                      np.int32, np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                gray = np.arange(0, 12, 2, dtype=dtype).reshape(2, 3)
                color = np.arange(24, dtype=dtype).reshape(2, 4, 3)[::-1, ::2, :]
                gray.setflags(write=False)
                color.setflags(write=False)
                result = instance.execute({"images": [gray, color]})
                self.assertEqual(result["__status_code"], 0)
                self.assertEqual(result["__status_message"], "运行成功")
                self.assertIsInstance(result["means"], list)
                self.assertEqual(result["means"], [5.0, 10.0])
                self.assertTrue(all(isinstance(value, float) for value in result["means"]))
                np.testing.assert_array_equal(gray, [[0, 2, 4], [6, 8, 10]])
                np.testing.assert_array_equal(color,
                                              [[[12, 13, 14], [18, 19, 20]],
                                               [[0, 1, 2], [6, 7, 8]]])
        self.assertEqual(instance.release()["__status_code"], 0)

    def test_counter_tool(self):
        mod = load_module_from_file(self.counter_py)
        instance = mod.CounterTool()
        init_res = instance.initialize({})
        self.assertEqual(init_res.get("__status_code"), 0)

        r1 = instance.execute({})
        self.assertEqual(r1.get("__status_code"), 0)
        self.assertEqual(r1.get("__status_message"), "运行成功")
        self.assertEqual(r1.get("count"), 1)

        r2 = instance.execute({})
        self.assertEqual(r2.get("__status_code"), 0)
        self.assertEqual(r2.get("__status_message"), "运行成功")
        self.assertEqual(r2.get("count"), 2)

        rel = instance.release()
        self.assertEqual(rel.get("__status_code"), 0)

    def test_roi_geometry_returns_three_layer_native_lists(self):
        declaration = json.loads((ROOT / "assets/tools/custom/Python/示例/ROI几何/ROI几何.json").read_text(encoding="utf-8"))
        execute = getattr(load_module_from_file(self.roi_py), declaration["entry"]["symbol"])
        images = [np.zeros((4, 5), dtype=np.uint8), np.zeros((3, 7, 3), dtype=np.uint8)]
        result = execute({"images": images})
        self.assertEqual(result["__status_code"], 0)
        self.assertEqual(result["__status_message"], "运行成功")
        self.assertEqual(result["rois"], [
            [[2.0, 0.0, 0.0, 5.0, 4.0, 0.0]],
            [[2.0, 0.0, 0.0, 7.0, 3.0, 0.0]],
        ])
        self.assertTrue(all(type(value) is float for group in result["rois"] for record in group for value in record))
        self.assertEqual(execute({"images": []})["rois"], [])
        self.assertEqual(execute({"images": [np.zeros((0, 5), dtype=np.uint8)]})["__status_code"], 1)
        self.assertIsNone(execute({"images": [np.zeros((0, 5), dtype=np.uint8)]})["rois"])

    def test_builtin_function_uses_its_json_entry_and_returns_native_lists(self):
        declaration = json.loads((ROOT / "assets/tools/builtin/Python/示例/图像尺寸.json").read_text(encoding="utf-8"))
        self.assertEqual(declaration["kind"], "python_function")
        entry = declaration["entry"]
        module = load_module_from_file(ROOT / "python/builtin" / entry["package"] / entry["file"])
        execute = getattr(module, entry["symbol"])
        images = [np.zeros((2, 3, 3), dtype=np.uint8), np.zeros((4, 1), dtype=np.float32)]
        result = execute({"images": images})
        self.assertEqual(result["__status_code"], 0)
        self.assertEqual(result["sizes"], [[2, 3], [4, 1]])
        self.assertTrue(all(type(value) is int for size in result["sizes"] for value in size))
        self.assertEqual(execute({"images": []})["sizes"], [])

if __name__ == "__main__":
    unittest.main()
