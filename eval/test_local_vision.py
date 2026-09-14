"""
eval/test_local_vision.py — Unit & Integration Tests for Native In-Process VLM Bridge (Moondream2)
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

from tools.local_vision import (
    scale_and_clamp_point,
    parse_moondream_point_output,
    get_action_coordinates,
    get_acceleration_device_and_dtype,
    verify_vision_dependencies,
    ground_and_click_target,
    HAS_VISION_DEPS,
)


class TestLocalVisionScalingAndParsing(unittest.TestCase):
    """Tests coordinate scaling and parsing from normalized spatial outputs."""

    def test_scale_and_clamp_normalized_midpoint(self):
        x, y = scale_and_clamp_point(0.5, 0.5, 1920, 1080)
        self.assertEqual(x, 960)
        self.assertEqual(y, 540)

    def test_scale_and_clamp_bounds(self):
        # Top-left corner
        x, y = scale_and_clamp_point(0.0, 0.0, 1920, 1080)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

        # Bottom-right corner (clamped to width - 1, height - 1)
        x, y = scale_and_clamp_point(1.0, 1.0, 1920, 1080)
        self.assertEqual(x, 1919)
        self.assertEqual(y, 1079)

    def test_scale_and_clamp_clamping_out_of_range(self):
        # Out of bounds high (pixel coordinates exceeding screen bounds)
        x, y = scale_and_clamp_point(2500, 1500, 1920, 1080)
        self.assertEqual(x, 1919)
        self.assertEqual(y, 1079)

        # Negative pixel coordinates
        x, y = scale_and_clamp_point(-50, -100, 1920, 1080)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_parse_moondream_point_output_points_list(self):
        raw = {"points": [{"x": 0.25, "y": 0.75}]}
        coords = parse_moondream_point_output(raw, image_width=1000, image_height=1000)
        self.assertEqual(coords, (250, 750))

    def test_parse_moondream_point_output_single_point_dict(self):
        raw = {"point": {"x": 0.1, "y": 0.2}}
        coords = parse_moondream_point_output(raw, image_width=800, image_height=600)
        self.assertEqual(coords, (80, 120))

    def test_parse_moondream_point_output_direct_xy_dict(self):
        raw = {"x": 0.6, "y": 0.4}
        coords = parse_moondream_point_output(raw, image_width=500, image_height=500)
        self.assertEqual(coords, (300, 200))

    def test_parse_moondream_point_output_raw_list(self):
        raw = [{"x": 0.3, "y": 0.8}]
        coords = parse_moondream_point_output(raw, image_width=100, image_height=100)
        self.assertEqual(coords, (30, 80))

    def test_parse_moondream_point_output_tuple_and_list_coords(self):
        coords = parse_moondream_point_output((0.5, 0.5), image_width=1000, image_height=800)
        self.assertEqual(coords, (500, 400))

    def test_parse_moondream_point_output_empty_or_invalid(self):
        self.assertIsNone(parse_moondream_point_output({}, 1920, 1080))
        self.assertIsNone(parse_moondream_point_output({"points": []}, 1920, 1080))
        self.assertIsNone(parse_moondream_point_output(None, 1920, 1080))
        self.assertIsNone(parse_moondream_point_output("invalid", 1920, 1080))


class TestLocalVisionInferenceAndExecution(unittest.TestCase):
    """Tests model inference dispatch and Win32 action routing."""

    def test_verify_vision_dependencies(self):
        self.assertTrue(HAS_VISION_DEPS)
        self.assertTrue(verify_vision_dependencies())

    def test_hardware_acceleration_detection(self):
        device, dtype = get_acceleration_device_and_dtype()
        self.assertIn(device, ("cuda", "cpu"))
        self.assertIsNotNone(dtype)

    def test_get_action_coordinates_with_mock_model(self):
        dummy_img = Image.new("RGB", (1280, 720), color="white")
        mock_model = MagicMock()
        mock_model.point.return_value = {"points": [{"x": 0.5, "y": 0.5}]}

        coords = get_action_coordinates(dummy_img, "Calculator icon", model=mock_model)
        self.assertIsNotNone(coords)
        self.assertEqual(coords, (640, 360))
        mock_model.point.assert_called_once_with(dummy_img, "Calculator icon")

    def test_get_action_coordinates_model_returns_empty(self):
        dummy_img = Image.new("RGB", (1000, 1000), color="black")
        mock_model = MagicMock()
        mock_model.point.return_value = {"points": []}

        coords = get_action_coordinates(dummy_img, "Non-existent element", model=mock_model)
        self.assertIsNone(coords)

    @patch("tools.screen_inspector.human_mouse_move")
    @patch("tools.screen_inspector.simulate_click")
    @patch("tools.screen_inspector.get_current_cursor_pos")
    def test_ground_and_click_target_simulated(self, mock_cur_pos, mock_sim_click, mock_mouse_move):
        dummy_img = Image.new("RGB", (1920, 1080), color="gray")
        mock_model = MagicMock()
        mock_model.point.return_value = {"points": [{"x": 0.2, "y": 0.3}]}
        mock_cur_pos.return_value = (100, 100)

        mock_sim_click.return_value.success = True
        mock_sim_click.return_value.action_log = "[SIMULATED CLICK]"

        res = ground_and_click_target(
            target_element="Settings button",
            duration=0.15,
            real_execution=False,
            model=mock_model,
            image_override=dummy_img,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["coordinates"], (384, 324))
        self.assertEqual(res["target"], "Settings button")
        mock_mouse_move.assert_called_once_with(100, 100, 384, 324, duration=0.15)
        mock_sim_click.assert_called_once()


if __name__ == "__main__":
    unittest.main()
