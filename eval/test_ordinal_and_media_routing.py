"""
eval/test_ordinal_and_media_routing.py — Unit Tests for Ordinal Grounding & Context-Aware Media Router
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("."))

from tools.vision_grounder import (
    ground_ordinal_query,
    parse_ordinal_from_query,
    select_ordinal_element,
    sort_elements_reading_order,
)
from tools.dispatcher import (
    resolve_intent_with_confidence,
    route_media_play,
)
from tools.system_dispatcher import route_media_play as sys_route_media_play


class DummyElement:
    def __init__(self, name: str, rect: tuple, control_type: str = "Button"):
        self.name = name
        self.rect = rect
        self.control_type = control_type

    def __repr__(self):
        return f"DummyElement({self.name}, rect={self.rect})"


class TestOrdinalAndMediaRouting(unittest.TestCase):

    def setUp(self):
        # 2x3 Grid of elements:
        # Row 1 (y ≈ 50): el1 at x=100, el2 at x=300, el3 at x=500
        # Row 2 (y ≈ 200): el4 at x=100, el5 at x=300, el6 at x=500
        self.el1 = DummyElement("Video 1", (100, 50, 250, 150))
        self.el2 = DummyElement("Video 2", (300, 52, 450, 152))
        self.el3 = DummyElement("Video 3", (500, 48, 650, 148))
        self.el4 = DummyElement("Video 4", (100, 200, 250, 300))
        self.el5 = DummyElement("Video 5", (300, 205, 450, 305))
        self.el6 = DummyElement("Video 6", (500, 198, 650, 298))

    def test_geometric_sorting_reading_order(self):
        """Verify elements in shuffled order are sorted top-to-bottom, left-to-right."""
        shuffled = [self.el6, self.el2, self.el5, self.el1, self.el4, self.el3]
        sorted_els = sort_elements_reading_order(shuffled, row_tolerance_px=15)

        expected_order = [self.el1, self.el2, self.el3, self.el4, self.el5, self.el6]
        self.assertEqual([e.name for e in sorted_els], [e.name for e in expected_order])

    def test_ordinal_parsing(self):
        """Verify parsing of numeric and word ordinals."""
        self.assertEqual(parse_ordinal_from_query("click the 1st video"), 1)
        self.assertEqual(parse_ordinal_from_query("click the second video"), 2)
        self.assertEqual(parse_ordinal_from_query("click the 3rd video"), 3)
        self.assertEqual(parse_ordinal_from_query("click the fourth video"), 4)
        self.assertEqual(parse_ordinal_from_query("select the last video"), -1)
        self.assertIsNone(parse_ordinal_from_query("click the video"))

    def test_select_ordinal_element(self):
        """Verify selecting 3rd video returns the 3rd element in reading order."""
        elements = [self.el6, self.el2, self.el5, self.el1, self.el4, self.el3]

        third_el = select_ordinal_element(elements, ordinal=3, filter_label="video")
        self.assertIsNotNone(third_el)
        self.assertEqual(third_el.name, "Video 3")

        first_el = select_ordinal_element(elements, ordinal=1)
        self.assertEqual(first_el.name, "Video 1")

        last_el = select_ordinal_element(elements, ordinal=-1)
        self.assertEqual(last_el.name, "Video 6")

        out_of_bounds = select_ordinal_element(elements, ordinal=99)
        self.assertIsNone(out_of_bounds)

    def test_ground_ordinal_query(self):
        """Verify full natural language ordinal grounding."""
        elements = [self.el6, self.el2, self.el5, self.el1, self.el4, self.el3]
        res = ground_ordinal_query("click the 3rd video", elements)
        self.assertTrue(res["found"])
        self.assertEqual(res["element"].name, "Video 3")
        self.assertEqual(res["ordinal"], 3)

    def test_media_disambiguation_pokemon_halts(self):
        """Verify 'Play Pokemon' triggers media disambiguation prompt in intent resolution."""
        action, is_conf, prompt = resolve_intent_with_confidence("Play Pokemon", "Play Query")
        self.assertEqual(action, "Play Query")
        self.assertFalse(is_conf)
        self.assertIn("local files", prompt)
        self.assertIn("YouTube videos", prompt)
        self.assertIn("YouTube Shorts", prompt)

    def test_media_router_query_disambiguation(self):
        """Verify route_media_play halts on unspecified destination and prompts user."""
        res = route_media_play("Play Pokemon")
        self.assertEqual(res["status"], "MEDIA_DISAMBIGUATION_REQUIRED")
        self.assertTrue(res["halted"])
        self.assertEqual(res["target"], "Pokemon")
        self.assertIn("YouTube Shorts", res["prompt"])

    def test_media_router_destination_launch_simulation(self):
        """Verify specifying YouTube, Shorts, or local files formats correct search launch."""
        # YouTube Videos
        res_yt = route_media_play("Play Pokemon", destination="YouTube videos", dry_run=True)
        self.assertEqual(res_yt["status"], "SIMULATED_MEDIA_LAUNCH")
        self.assertFalse(res_yt["halted"])
        self.assertIn("youtube.com/results?search_query=Pokemon", res_yt["url"])

        # YouTube Shorts
        res_shorts = route_media_play("Play Pokemon", destination="YouTube Shorts", dry_run=True)
        self.assertIn("shorts", res_shorts["url"])

        # Local Files
        res_local = route_media_play("Play Pokemon", destination="local files", dry_run=True)
        self.assertIn("file:///search?query=Pokemon", res_local["url"])

    def test_system_dispatcher_facade(self):
        """Verify tools/system_dispatcher.py exports route_media_play."""
        res = sys_route_media_play("Play Pokemon")
        self.assertEqual(res["status"], "MEDIA_DISAMBIGUATION_REQUIRED")


if __name__ == "__main__":
    unittest.main()
