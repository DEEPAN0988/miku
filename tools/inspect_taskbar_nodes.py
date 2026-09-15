"""
tools/inspect_taskbar_nodes.py — Diagnostic tool to dump active UIAutomation node names
"""
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from ui_inspector import TreeInspector

def main():
    inspector = TreeInspector()
    nodes = inspector.inspect_tree()
    print(f"Total UIA nodes found on screen: {len(nodes)}")
    for i, n in enumerate(nodes[:30], 1):
        print(f"[{i:02d}] Name: '{n.name}' | AutoId: '{n.automation_id}' | Type: {n.control_type} | Center: {n.center}")

if __name__ == "__main__":
    main()
