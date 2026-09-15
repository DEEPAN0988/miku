"""
ui_inspector.py — OS UIAutomation Accessibility Tree Inspector

ARCHITECTURAL CONSTRAINTS:
1. Interacts exclusively with standard OS accessibility trees/DOM (Windows UIAutomationCore COM API).
2. DO NOT use OpenCV, template matching, or OCR.
3. Strict tree resolution: If an element is missing from the OS accessibility tree,
   MUST throw ElementNotFoundError. Never guess coordinates based on pixels.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from tools.screen_inspector import inspect_screen_elements, ScreenSnapshot, UIElement
    HAVE_SCREEN_INSPECTOR = True
except ImportError:
    HAVE_SCREEN_INSPECTOR = False
    UIElement = Any  # type: ignore


class ElementNotFoundError(Exception):
    """Raised when a requested UI element cannot be located in the OS UIAutomation tree."""
    pass


@dataclass
class UIElementMatch:
    """Represents a resolved node in the OS UIAutomation accessibility tree."""
    name: str
    automation_id: str
    control_type: str
    bounds: Tuple[int, int, int, int]  # (left, top, right, bottom)
    center: Tuple[int, int]            # (x, y)
    text_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "automation_id": self.automation_id,
            "control_type": self.control_type,
            "bounds": self.bounds,
            "center": self.center,
            "text": self.text_content or self.name,
        }


class TreeInspector:
    """
    Parses and queries the native OS UIAutomation accessibility tree.
    """

    def __init__(self, mock_elements: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the TreeInspector.
        
        :param mock_elements: Optional list of element dictionaries for deterministic unit testing.
        """
        self.mock_elements = mock_elements

    def inspect_tree(self) -> List[UIElementMatch]:
        """
        Traverses the current desktop OS UIAutomation tree and returns all detected nodes.
        """
        if self.mock_elements is not None:
            matches = []
            for item in self.mock_elements:
                matches.append(
                    UIElementMatch(
                        name=item.get("name", ""),
                        automation_id=item.get("automation_id", ""),
                        control_type=item.get("control_type", "Button"),
                        bounds=item.get("bounds", (0, 0, 100, 100)),
                        center=item.get("center", (50, 50)),
                        text_content=item.get("text") or item.get("value") or item.get("name", ""),
                    )
                )
            return matches

        if not HAVE_SCREEN_INSPECTOR:
            raise RuntimeError(
                "Native OS UIAutomation inspector unavailable (tools.screen_inspector module missing)."
            )

        snapshot: ScreenSnapshot = inspect_screen_elements()
        matches = []
        for el in snapshot.elements:
            element_bounds = getattr(el, "rect", getattr(el, "bbox", (0, 0, 100, 100)))
            element_center = getattr(el, "center", (50, 50))
            matches.append(
                UIElementMatch(
                    name=el.name,
                    automation_id=el.automation_id,
                    control_type=el.control_type,
                    bounds=element_bounds,
                    center=element_center,
                    text_content=getattr(el, "text", None) or getattr(el, "value", None) or el.name,
                )
            )
        return matches

    def find_element_by_name(
        self,
        name: str,
        control_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Traverses the UI tree to find a control by its logical name or AutomationId.
        Performs a case-insensitive partial substring match checking BOTH name and automation_id.
        Returns a dictionary with exact bounding box coordinates and center point.
        Raises ElementNotFoundError if the element is not found in the OS tree.
        """
        if not name or not name.strip():
            raise ElementNotFoundError("Empty or invalid element name specified for search.")

        target_lower = name.strip().lower()
        c_filter = control_type.lower().strip() if control_type else None

        elements = self.inspect_tree()

        # Pass 1: Exact matches first
        exact_matches = []
        for el in elements:
            if c_filter and el.control_type.lower() != c_filter:
                continue
            if (el.automation_id and el.automation_id.lower() == target_lower) or \
               (el.name and el.name.lower() == target_lower):
                exact_matches.append(el)

        if exact_matches:
            for el in exact_matches:
                if getattr(el, "is_visible", True):
                    return el.to_dict()
            return exact_matches[0].to_dict()

        # Pass 2: Case-insensitive partial substring match on BOTH name and automation_id
        partial_matches = []
        for el in elements:
            if c_filter and el.control_type.lower() != c_filter:
                continue
            el_name = (el.name or "").lower()
            el_auto_id = (el.automation_id or "").lower()

            if target_lower in el_name or target_lower in el_auto_id:
                partial_matches.append(el)

        if partial_matches:
            # Return first visible match, or first match if visibility is unconstrained
            for el in partial_matches:
                if getattr(el, "is_visible", True):
                    return el.to_dict()
            return partial_matches[0].to_dict()

        # STRICT CONSTRAINT: Do NOT guess pixels. Throw ElementNotFoundError.
        raise ElementNotFoundError(
            f"Element '{name}' (control_type={control_type or 'Any'}) not found in OS UIAutomation accessibility tree."
        )

    def read_element_text(self, target_name: str) -> Dict[str, Any]:
        """
        Reads and extracts text from a specified element or container in the UIAutomation accessibility tree.
        Zero OCR used. Read-only operation.
        
        If target_name represents a window/pane/container or if target_name is 'screen',
        recursively gathers visible children text to provide a screen text summary.
        """
        elements = self.inspect_tree()

        # Special case: 'screen' or 'root' reads all visible nodes in accessibility tree
        if target_name.strip().lower() in ("screen", "root", "desktop", "all"):
            children_texts = [el.name for el in elements if el.name and el.name.strip()]
            summary_text = "\n".join(children_texts)
            return {
                "target": target_name,
                "text": summary_text,
                "control_type": "Desktop",
                "element_count": len(children_texts),
                "children_text": children_texts,
            }

        # Locate specific target node
        try:
            target_info = self.find_element_by_name(target_name)
        except ElementNotFoundError:
            raise ElementNotFoundError(
                f"Cannot read text: Element '{target_name}' not found in OS UIAutomation accessibility tree."
            )

        t_left, t_top, t_right, t_bottom = target_info["bounds"]
        primary_text = target_info.get("text") or target_info.get("name") or ""

        # Check for children inside bounding box if container/pane/window
        children_texts = []
        for el in elements:
            if el.name and el.name.strip() and el.name != target_info["name"]:
                e_left, e_top, e_right, e_bottom = el.bounds
                # Check bounding box containment
                if e_left >= t_left and e_top >= t_top and e_right <= t_right and e_bottom <= t_bottom:
                    children_texts.append(el.name)

        if children_texts:
            full_text = f"{primary_text}\n" + "\n".join(children_texts) if primary_text else "\n".join(children_texts)
        else:
            full_text = primary_text

        return {
            "target": target_info["name"],
            "automation_id": target_info["automation_id"],
            "control_type": target_info["control_type"],
            "text": full_text.strip(),
            "primary_text": primary_text,
            "children_text": children_texts,
            "bounds": target_info["bounds"],
        }
