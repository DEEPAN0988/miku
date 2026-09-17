import unittest
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.engine import PlannerEngine


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.test_dir, "test.db")
        self.planner = PlannerEngine(db_path=self.test_db)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_planner_tasks(self):
        # 1. Create task
        t = self.planner.create_task(title="Test Task", priority="high")
        self.assertTrue(t["success"])
        self.assertGreater(t["task_id"], 0)

        # 2. List tasks
        pending = self.planner.get_tasks(status="pending")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["title"], "Test Task")

        # 3. Complete task
        ok = self.planner.complete_task(t["task_id"])
        self.assertTrue(ok)
        self.assertEqual(len(self.planner.get_tasks(status="pending")), 0)

    def test_planner_day_plan_and_goals(self):
        # Goal breakdown
        subtasks = self.planner.plan_goal("training for marathon")
        self.assertEqual(len(subtasks), 3)

        # Schedule event
        event = self.planner.schedule_event(title="Morning Workout", start_time="2026-09-18 08:00")
        self.assertTrue(event["success"])

        plan = self.planner.get_day_plan(target_date="2026-09-18")
        self.assertEqual(plan["total_events"], 1)


if __name__ == "__main__":
    unittest.main()
