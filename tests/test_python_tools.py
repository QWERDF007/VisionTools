#!/usr/bin/env python3
"""Public lifecycle and input/output contracts for Python example tools.

Every tool answers with the shipped protocol: ``execute`` returns
``{"status": {...}, "outputs": {...}}`` and ``initialize``/``release`` return the status dict
alone. These tests read the same channels the VisionRT adapter reads, so a tool cannot pass here
and fail in the host.
"""

import importlib
import json
import pathlib
import sys
import types
import unittest
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUTHOR_SDK = ROOT / "python/author_sdk"


def load_module_from_file(file_path: pathlib.Path):
    """Loads a tool the way the host does: its directory is a package root and the entry file is a
    module inside it, so relative imports resolve without touching ``sys.path``. The author SDK is
    resolved from its single repository source; a shipped package carries the same file inside the
    tool directory, which is exactly the layout the packager writes."""
    root_name = "vt_tool_" + file_path.parent.name
    if root_name not in sys.modules:
        package = types.ModuleType(root_name)
        package.__path__ = [str(file_path.parent), str(AUTHOR_SDK)]
        sys.modules[root_name] = package
    return importlib.import_module(f"{root_name}.{file_path.stem}")


def status(result):
    """The one status an execution reports."""
    return {
        "code": result["outputs"]["__status_code"],
        "message": result["outputs"]["__status_message"],
        "detail": result.get("detail", ""),
    }


def outputs(result):
    """The declared business ports, never the status."""
    return {k: v for k, v in result["outputs"].items() if not k.startswith("__status_")}


class TestPythonToolContracts(unittest.TestCase):
    def setUp(self):
        mean = json.loads((ROOT / "assets/tools/custom/Python/示例/计算图像均值/计算图像均值.json").read_text(encoding="utf-8"))["entry"]
        counter = json.loads((ROOT / "assets/tools/custom/Python/示例/调用计数器/调用计数器.json").read_text(encoding="utf-8"))["entry"]
        roi = json.loads((ROOT / "assets/tools/custom/Python/示例/ROI几何/ROI几何.json").read_text(encoding="utf-8"))["entry"]
        add = json.loads((ROOT / "assets/tools/custom/Python/示例/图像加一/图像加一.json").read_text(encoding="utf-8"))["entry"]
        self.mean_py = ROOT / "python/custom" / mean["package"] / mean["file"]
        self.counter_py = ROOT / "python/custom" / counter["package"] / counter["file"]
        self.roi_py = ROOT / "python/custom" / roi["package"] / roi["file"]
        self.image_add_py = ROOT / "python/custom" / add["package"] / add["file"]
        self.assertTrue(self.mean_py.exists(), f"mean.py not found at {self.mean_py}")
        self.assertTrue(self.counter_py.exists(), f"counter.py not found at {self.counter_py}")
        self.assertTrue(self.roi_py.exists(), f"ROI tool.py not found at {self.roi_py}")
        self.assertTrue(self.image_add_py.exists(), f"image add tool.py not found at {self.image_add_py}")

    def test_lifecycle_returns_a_bare_status(self):
        instance = load_module_from_file(self.mean_py).ToolInstance()
        for answer in (instance.initialize({}), instance.release()):
            self.assertEqual(set(answer), {"code", "message", "detail"})
            self.assertEqual(answer["code"], 0)
        counter = load_module_from_file(self.counter_py).CounterTool()
        self.assertEqual(counter.initialize({}), {"code": 0, "message": "", "detail": ""})
        self.assertEqual(counter.release(), {"code": 0, "message": "", "detail": ""})

    def test_mean_missing_input(self):
        mod = load_module_from_file(self.mean_py)
        instance = mod.ToolInstance()
        instance.initialize({})

        for inputs in ({}, {"images": None}):
            with self.subTest(inputs=inputs):
                result = instance.execute(inputs)
                self.assertEqual(status(result)["code"], 1001)
                self.assertIn("输入端口 'images' 缺失", status(result)["message"])
                self.assertIsNone(outputs(result)["means"])

        instance.release()

    def test_mean_rejects_non_batch_inputs(self):
        instance = load_module_from_file(self.mean_py).ToolInstance()
        instance.initialize({})
        image = np.ones((2, 3), dtype=np.uint8)
        for images in (image, (image,), {"image": image}, 7):
            with self.subTest(container=type(images).__name__):
                result = instance.execute({"images": images})
                self.assertEqual(status(result)["code"], 1002)
                self.assertIn("输入端口 'images'", status(result)["message"])
                self.assertIn("列表", status(result)["message"])
                self.assertIsNone(outputs(result)["means"])

    def test_mean_rejects_non_image_elements(self):
        instance = load_module_from_file(self.mean_py).ToolInstance()
        instance.initialize({})
        image = np.ones((2, 3), dtype=np.uint8)
        for invalid in (np.float32(2), "bad", [[1, 2]], None):
            with self.subTest(element=type(invalid).__name__):
                result = instance.execute({"images": [image, invalid]})
                self.assertEqual(status(result)["code"], 1002)
                self.assertIn("输入端口 'images'", status(result)["message"])
                self.assertIn("批次索引 1", status(result)["message"])
                self.assertIn("ndarray", status(result)["message"])
                self.assertIsNone(outputs(result)["means"])

    def test_mean_rejects_invalid_image_shapes(self):
        instance = load_module_from_file(self.mean_py).ToolInstance()
        instance.initialize({})
        valid = np.ones((2, 3), dtype=np.uint8)
        for shape in ((), (3,), (1, 2, 3, 4), (0, 3), (2, 0), (2, 3, 0)):
            with self.subTest(shape=shape):
                result = instance.execute({"images": [valid, np.zeros(shape, dtype=np.uint8)]})
                self.assertEqual(status(result)["code"], 1002)
                self.assertIn("输入端口 'images'", status(result)["message"])
                self.assertIn("批次索引 1", status(result)["message"])
                self.assertIn("图像形状", status(result)["message"])
                self.assertIn(str(shape), status(result)["message"])
                self.assertIsNone(outputs(result)["means"])

    def test_mean_rejects_non_real_pixels(self):
        instance = load_module_from_file(self.mean_py).ToolInstance()
        instance.initialize({})
        image = np.ones((2, 3), dtype=np.uint8)
        for dtype in (np.str_, object, np.bool_, np.complex64):
            with self.subTest(dtype=dtype):
                invalid = np.full((2, 3), 2, dtype=dtype)
                result = instance.execute({"images": [image, invalid]})
                self.assertEqual(status(result)["code"], 1002)
                self.assertIn("输入端口 'images'", status(result)["message"])
                self.assertIn("批次索引 1", status(result)["message"])
                self.assertIn("实数数值", status(result)["message"])
                self.assertIn(str(invalid.dtype), status(result)["message"])
                self.assertIsNone(outputs(result)["means"])

    def test_mean_empty_batch(self):
        mod = load_module_from_file(self.mean_py)
        instance = mod.ToolInstance()
        instance.initialize({})

        result = instance.execute({"images": []})
        self.assertEqual(status(result)["code"], 0)
        self.assertEqual(status(result)["message"], "运行成功")
        means = outputs(result)["means"]
        self.assertIsInstance(means, list)
        self.assertEqual(means, [])

    def test_mean_ragged_inputs_not_modified(self):
        mod = load_module_from_file(self.mean_py)
        instance = mod.ToolInstance()
        instance.initialize({})

        img1 = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        img1.setflags(write=False)  # Read-only view
        img2 = np.array([[6.0, 7.0], [7.0, 8.0]], dtype=np.uint8)
        img2.setflags(write=False)  # Read-only view

        result = instance.execute({"images": [img1, img2]})
        self.assertEqual(status(result)["code"], 0)
        self.assertEqual(status(result)["message"], "运行成功")
        means = outputs(result)["means"]
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
                self.assertEqual(status(result)["code"], 0)
                self.assertEqual(status(result)["message"], "运行成功")
                self.assertIsInstance(outputs(result)["means"], list)
                self.assertEqual(outputs(result)["means"], [5.0, 10.0])
                self.assertTrue(all(isinstance(value, float) for value in outputs(result)["means"]))
                np.testing.assert_array_equal(gray, [[0, 2, 4], [6, 8, 10]])
                np.testing.assert_array_equal(color,
                                              [[[12, 13, 14], [18, 19, 20]],
                                               [[0, 1, 2], [6, 7, 8]]])
        self.assertEqual(instance.release()["code"], 0)

    def test_counter_tool(self):
        mod = load_module_from_file(self.counter_py)
        instance = mod.CounterTool()
        init_res = instance.initialize({})
        self.assertEqual(init_res["code"], 0)

        r1 = instance.execute({})
        self.assertEqual(status(r1)["code"], 0)
        self.assertEqual(status(r1)["message"], "运行成功")
        self.assertEqual(outputs(r1)["count"], 1)

        r2 = instance.execute({})
        self.assertEqual(status(r2)["code"], 0)
        self.assertEqual(status(r2)["message"], "运行成功")
        self.assertEqual(outputs(r2)["count"], 2)

        rel = instance.release()
        self.assertEqual(rel["code"], 0)

    def test_roi_geometry_returns_three_layer_native_lists(self):
        declaration = json.loads((ROOT / "assets/tools/custom/Python/示例/ROI几何/ROI几何.json").read_text(encoding="utf-8"))
        execute = getattr(load_module_from_file(self.roi_py), declaration["entry"]["symbol"])
        images = [np.zeros((4, 5), dtype=np.uint8), np.zeros((3, 7, 3), dtype=np.uint8)]
        result = execute({"images": images})
        self.assertEqual(status(result)["code"], 0)
        self.assertEqual(status(result)["message"], "运行成功")
        self.assertEqual(outputs(result)["rois"], [
            [[2.0, 0.0, 0.0, 5.0, 4.0, 0.0]],
            [[2.0, 0.0, 0.0, 7.0, 3.0, 0.0]],
        ])
        self.assertTrue(all(type(value) is float
                            for group in outputs(result)["rois"]
                            for record in group
                            for value in record))
        self.assertEqual(outputs(execute({"images": []}))["rois"], [])
        rejected = execute({"images": [np.zeros((0, 5), dtype=np.uint8)]})
        self.assertEqual(status(rejected)["code"], 1002)
        self.assertIsNone(outputs(rejected)["rois"])

    def test_image_add_distinguishes_copy_and_inplace(self):
        declaration = json.loads((ROOT / "assets/tools/custom/Python/示例/图像加一/图像加一.json").read_text(encoding="utf-8"))
        execute = getattr(load_module_from_file(self.image_add_py), declaration["entry"]["symbol"])
        source = np.full((2, 3), 10, dtype=np.uint8)
        copied = execute({"images": [source], "mode": "copy"})
        self.assertEqual(status(copied)["code"], 0)
        self.assertTrue(np.array_equal(source, np.full((2, 3), 10, dtype=np.uint8)))
        self.assertTrue(np.array_equal(outputs(copied)["images"][0], np.full((2, 3), 11, dtype=np.uint8)))

        original = np.full((2, 3), 20, dtype=np.uint8)
        inplace = execute({"images": [original], "mode": "inplace"})
        self.assertEqual(status(inplace)["code"], 0)
        self.assertTrue(np.array_equal(original, np.full((2, 3), 21, dtype=np.uint8)))
        self.assertIs(outputs(inplace)["images"][0], original)

        self.assertEqual(outputs(execute({"images": [], "mode": "copy"}))["images"], [])
        invalid = execute({"images": [source], "mode": "other"})
        self.assertEqual(status(invalid)["code"], 1003)
        self.assertIsNone(outputs(invalid)["images"])

    def test_image_add_honours_upstream_status(self):
        declaration = json.loads((ROOT / "assets/tools/custom/Python/示例/图像加一/图像加一.json").read_text(encoding="utf-8"))
        execute = getattr(load_module_from_file(self.image_add_py), declaration["entry"]["symbol"])
        source = np.full((2, 3), 10, dtype=np.uint8)
        skipped = execute({"images": [source], "mode": "inplace", "upstream_status": 7})
        self.assertEqual(status(skipped)["code"], 7)
        self.assertEqual(status(skipped)["message"], "上游未选中本分支，工具未执行")
        self.assertIsNone(outputs(skipped)["images"])
        self.assertTrue(np.array_equal(source, np.full((2, 3), 10, dtype=np.uint8)),
                        "a non-zero upstream status must not modify the source image")
        recovered = execute({"images": [source], "mode": "copy", "upstream_status": 0})
        self.assertEqual(status(recovered)["code"], 0)
        self.assertTrue(np.array_equal(outputs(recovered)["images"][0], np.full((2, 3), 11, dtype=np.uint8)))
        # A non-integer upstream status is a mistyped port, not a missing one.
        self.assertEqual(status(execute({"images": [source], "upstream_status": True}))["code"], 1002)
        none_zero = execute({"images": None, "mode": "copy", "upstream_status": 0})
        self.assertEqual(status(none_zero)["code"], 1001)
        self.assertIn("缺失", status(none_zero)["message"])
        none_seven = execute({"images": None, "mode": "copy", "upstream_status": 7})
        self.assertEqual(status(none_seven)["code"], 7)
        self.assertEqual(status(none_seven)["message"], "上游未选中本分支，工具未执行")
        bad_type_zero = execute({"images": 12345, "mode": "copy", "upstream_status": 0})
        self.assertEqual(status(bad_type_zero)["code"], 1002)
        bad_type_seven = execute({"images": 12345, "mode": "copy", "upstream_status": 7})
        self.assertEqual(status(bad_type_seven)["code"], 1002)

    def test_mean_keeps_measuring_after_upstream_failure(self):
        instance = load_module_from_file(self.mean_py).ToolInstance()
        instance.initialize({})
        result = instance.execute({"images": [np.full((2, 2), 3.0)], "upstream_status": 7})
        self.assertEqual(status(result)["code"], 0)
        self.assertEqual(outputs(result)["means"], [3.0])

    def test_builtin_function_uses_its_json_entry_and_returns_native_lists(self):
        declaration = json.loads((ROOT / "assets/tools/builtin/Python/示例/图像尺寸.json").read_text(encoding="utf-8"))
        self.assertEqual(declaration["kind"], "python_function")
        entry = declaration["entry"]
        module = load_module_from_file(ROOT / "python/builtin" / entry["package"] / entry["file"])
        execute = getattr(module, entry["symbol"])
        images = [np.zeros((2, 3, 3), dtype=np.uint8), np.zeros((4, 1), dtype=np.float32)]
        result = execute({"images": images})
        self.assertEqual(status(result)["code"], 0)
        self.assertEqual(outputs(result)["sizes"], [[2, 3], [4, 1]])
        self.assertTrue(all(type(value) is int for size in outputs(result)["sizes"] for value in size))
        self.assertEqual(outputs(execute({"images": []}))["sizes"], [])


if __name__ == "__main__":
    unittest.main()
