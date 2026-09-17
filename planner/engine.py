"""
Planner & Scheduler Engine for Miku.
Handles task planning, day scheduling, goal decomposition, and schedule queries.
Rule and logic based - no ML required per PRD Section 5.5.
"""

from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from .database import (
    init_db, add_task, list_tasks, update_task_status,
    add_calendar_event, list_calendar_events
)


class PlannerEngine:
    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            init_db(db_path)
        else:
            init_db()
        self.db_path = db_path

    def create_task(self, title: str, description: str = "", priority: str = "medium", due_date: Optional[str] = None) -> Dict[str, Any]:
        task_id = add_task(title, description, priority, due_date, db_path=self.db_path) if self.db_path else add_task(title, description, priority, due_date)
        return {
            "success": True,
            "task_id": task_id,
            "title": title,
            "priority": priority,
            "due_date": due_date
        }

    def get_tasks(self, status: Optional[str] = "pending") -> List[Dict[str, Any]]:
        return list_tasks(status=status, db_path=self.db_path) if self.db_path else list_tasks(status=status)

    def complete_task(self, task_id: int) -> bool:
        return update_task_status(task_id, "completed", db_path=self.db_path) if self.db_path else update_task_status(task_id, "completed")

    def schedule_event(self, title: str, start_time: str, end_time: Optional[str] = None, location: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
        event_id = add_calendar_event(title, start_time, end_time, location, description, db_path=self.db_path) if self.db_path else add_calendar_event(title, start_time, end_time, location, description)
        return {
            "success": True,
            "event_id": event_id,
            "title": title,
            "start_time": start_time,
            "end_time": end_time
        }

    def get_day_plan(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a complete schedule and task summary for the specified date (default: today).
        """
        if not target_date:
            target_date = date.today().isoformat()
        
        events = list_calendar_events(date_str=target_date, db_path=self.db_path) if self.db_path else list_calendar_events(date_str=target_date)
        pending_tasks = self.get_tasks(status="pending")
        
        # Breakdown into a structured daytime itinerary
        morning_events = [e for e in events if "06:00" <= e["start_time"][-5:] < "12:00"]
        afternoon_events = [e for e in events if "12:00" <= e["start_time"][-5:] < "17:00"]
        evening_events = [e for e in events if e["start_time"][-5:] >= "17:00"]

        return {
            "date": target_date,
            "total_events": len(events),
            "events": events,
            "pending_tasks": pending_tasks,
            "summary": (
                f"Schedule for {target_date}: You have {len(events)} scheduled event(s) "
                f"and {len(pending_tasks)} pending task(s)."
            )
        }

    def plan_goal(self, goal_text: str) -> List[Dict[str, Any]]:
        """
        Rule-based goal breakdown into sub-tasks (e.g., 'training for marathon', 'study Python').
        """
        lower = goal_text.lower()
        subtasks = []
        if "marathon" in lower or "run" in lower or "fitness" in lower:
            subtasks = [
                {"title": "Morning 5km endurance run", "priority": "high"},
                {"title": "Hydration and protein recovery meal", "priority": "medium"},
                {"title": "Stretching & mobility workout", "priority": "low"}
            ]
        elif "study" in lower or "learn" in lower or "read" in lower:
            subtasks = [
                {"title": "Read core reference chapter for 45 minutes", "priority": "high"},
                {"title": "Hands-on coding exercise / practical drill", "priority": "high"},
                {"title": "Review summary notes and flashcards", "priority": "medium"}
            ]
        elif "essay" in lower or "write" in lower or "report" in lower:
            subtasks = [
                {"title": "Research background material and outline structure", "priority": "high"},
                {"title": "Draft main content sections", "priority": "high"},
                {"title": "Proofread and polish final text", "priority": "medium"}
            ]
        else:
            subtasks = [
                {"title": f"Define clear milestones for: {goal_text}", "priority": "high"},
                {"title": f"Execute first action item for: {goal_text}", "priority": "high"},
                {"title": "Review progress and adjust timeline", "priority": "medium"}
            ]
        
        created = []
        for st in subtasks:
            res = self.create_task(title=st["title"], priority=st["priority"])
            created.append(res)
        return created
