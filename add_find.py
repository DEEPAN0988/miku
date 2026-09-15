import sys
with open(r'c:\miku\tools\screen_inspector.py', 'a', encoding='utf-8') as f:
    f.write('''
def find_element_bounds(target_name: str) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]:
    """
    Enumerates visible UI elements in the active/foreground window and finds an element 
    matching target_name.
    
    Returns:
        (rect, center) where rect is (left, top, right, bottom) and center is (cx, cy),
        or None if no match is found.
    """
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    if not hwnd:
        return None
        
    snapshot = inspect_screen_elements(hwnd=hwnd, max_elements=400)
    
    target_name_lower = target_name.lower().strip()
    
    # 1. Exact match on name or automation_id
    for el in snapshot.elements:
        if not el.is_enabled: continue
        if el.name.lower().strip() == target_name_lower or el.automation_id.lower().strip() == target_name_lower:
            return el.rect, el.center
            
    # 2. Fuzzy match
    for el in snapshot.elements:
        if not el.is_enabled: continue
        if target_name_lower in el.name.lower() or target_name_lower in el.automation_id.lower() or target_name_lower in el.class_name.lower():
            return el.rect, el.center
            
    return None
''')
