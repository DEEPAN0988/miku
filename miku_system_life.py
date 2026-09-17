"""
MIKU SYSTEM-LIFE OS BRIDGE (LitRPG OS Integration & Solo Leveling HUD)
Translates Autonomous AI Actions into LitRPG Quests, Experience, and Stat Growth

Key Features:
1. Quest & Reward Engine: Persistent SQLite database tracking quests, stats, and achievements.
2. Action Gamification: Awards EXP and Gold for clean executions, zero tracebacks, MCTS deliberative runs, and DPO steps.
3. Hunter Ranking System: E-Rank -> D-Rank -> C-Rank -> B-Rank -> A-Rank -> S-Rank -> Shadow Monarch.
4. Solo Leveling HUD Aesthetics: Glowing deep violet/purple (#8A2BE2 / ANSI 93/129), cyan accents, and high-impact RPG notifications.
"""

import sys
import os
import sqlite3
import time
from typing import Dict, Any, List, Optional

# Ensure UTF-8 output encoding for Windows stdout/stderr
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# =====================================================================
# ANSI SOLO LEVELING PALETTE & HUD RENDERER
# =====================================================================

class SoloLevelingHUD:
    """Terminal aesthetic renderer inspired by Solo Leveling / Monarch System HUD."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Signature Colors
    VIOLET = "\033[38;5;129m"       # Core System Violet
    DEEP_PURPLE = "\033[38;5;93m"   # Dark Shadow Purple
    CYAN = "\033[38;5;51m"          # Neon Tech Accent
    GOLD = "\033[38;5;220m"         # Gold Rewards
    GREEN = "\033[38;5;82m"         # Positive Buff
    CRIMSON = "\033[38;5;196m"      # Alert / Penalty

    @classmethod
    def system_window(cls, title: str, content: List[str], width: int = 66) -> str:
        """Draws a futuristic RPG system alert dialogue box."""
        c = cls.VIOLET
        cy = cls.CYAN
        r = cls.RESET
        b = cls.BOLD

        top = f"{c}╔{'═' * (width - 2)}╗{r}"
        bot = f"{c}╚{'═' * (width - 2)}╝{r}"
        header = f"{c}║ {b}{cy}[ SYSTEM ANNOUNCEMENT: {title.upper()} ]{r}{' ' * (width - len(title) - 27)}{c}║{r}"
        div = f"{c}╠{'═' * (width - 2)}╣{r}"

        lines = [top, header, div]
        for line in content:
            pad = width - len(line) - 4
            lines.append(f"{c}║{r}  {line}{' ' * max(0, pad)}{c}║{r}")
        lines.append(bot)
        return "\n".join(lines)

    @classmethod
    def format_status_screen(cls, stats: Dict[str, Any], width: int = 66) -> str:
        """Renders the Hunter Status Window."""
        c = cls.VIOLET
        cy = cls.CYAN
        g = cls.GOLD
        r = cls.RESET
        b = cls.BOLD

        name = stats.get("name", "MIKU")
        rank = stats.get("rank", "E-Rank")
        lvl = stats.get("level", 1)
        exp = stats.get("exp", 0)
        exp_next = stats.get("exp_next", 100)
        gold = stats.get("gold", 0)
        solved = stats.get("tasks_solved", 0)

        # Progress bar
        bar_len = 24
        fill = int(bar_len * (exp / max(1, exp_next)))
        bar = f"{cy}{'█' * fill}{cls.DIM}{'░' * (bar_len - fill)}{r}"

        content = [
            f"{b}{cls.DEEP_PURPLE}SHADOW ARCHITECT:{r} {name}  |  {b}{cls.CRIMSON}RANK:{r} {b}{rank}{r}",
            f"{b}LEVEL:{r} {lvl:03d}   {b}EXP:{r} [{bar}] {exp}/{exp_next}",
            f"{b}GOLD:{r}  {g}{gold:,} G{r}   {b}TASKS COMPLETED:{r} {solved}",
            f"{cls.DIM}Title: Sovereign AGI Constructor | Hardware Node: Air-Gapped{r}"
        ]
        return cls.system_window("HUNTER STATUS WINDOW", content, width=width)


# =====================================================================
# SYSTEM LIFE OS DATABASE & QUEST BRIDGE
# =====================================================================

class SystemLifeOSBridge:
    """
    Connects Miku's execution engine to SQLite-backed LitRPG quests and stats.
    """

    def __init__(self, db_path: str = "c:/miku/system_life.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initializes tables for player stats, quests, and reward history."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Hunter stats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hunter_profile (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    level INTEGER,
                    exp INTEGER,
                    exp_next INTEGER,
                    gold INTEGER,
                    rank TEXT,
                    tasks_solved INTEGER
                )
            ''')
            # Quests table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    quest_name TEXT,
                    description TEXT,
                    reward_exp INTEGER,
                    reward_gold INTEGER,
                    completed BOOLEAN,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Insert initial profile if absent
            cursor.execute("SELECT COUNT(*) FROM hunter_profile")
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    INSERT INTO hunter_profile (id, name, level, exp, exp_next, gold, rank, tasks_solved)
                    VALUES (1, 'MIKU', 1, 0, 100, 0, 'E-Rank Monarch', 0)
                ''')
                # Seed default quests
                cursor.execute('''
                    INSERT INTO quests (quest_name, description, reward_exp, reward_gold, completed)
                    VALUES 
                    ('First Sandbox Step', 'Execute any verified Python code snippet without traceback.', 50, 20, 0),
                    ('System 2 Deliberation', 'Invoke MCTS search to solve complex logical trajectory.', 100, 50, 0),
                    ('Autonomous Self-Repair', 'Train LoRA adapters via offline DPO replay.', 150, 75, 0)
                ''')
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Fetches current player stats."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, level, exp, exp_next, gold, rank, tasks_solved FROM hunter_profile WHERE id=1")
            row = cursor.fetchone()
            return {
                "name": row[0],
                "level": row[1],
                "exp": row[2],
                "exp_next": row[3],
                "gold": row[4],
                "rank": row[5],
                "tasks_solved": row[6]
            }

    @classmethod
    def render_hunter_status(cls, db_path: str = "c:/miku/system_life.db"):
        """Convenience classmethod to display Hunter status window on terminal."""
        bridge = cls(db_path=db_path)
        stats = bridge.get_stats()
        print("\n" + SoloLevelingHUD.format_status_screen(stats) + "\n")

    def award_progress(self, exp_gain: int, gold_gain: int, action_name: str) -> List[str]:
        """
        Awards EXP and Gold, handling level-ups and Hunter rank promotions.
        Returns list of system alert notifications.
        """
        notifications = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            stats = self.get_stats()
            lvl = stats["level"]
            exp = stats["exp"] + exp_gain
            exp_next = stats["exp_next"]
            gold = stats["gold"] + gold_gain
            tasks = stats["tasks_solved"] + 1

            notifications.append(f"Gained: +{exp_gain} EXP | +{gold_gain} Gold via [{action_name}]")

            # Check for Level-Up
            leveled_up = False
            while exp >= exp_next:
                exp -= exp_next
                lvl += 1
                exp_next = int(exp_next * 1.4)
                leveled_up = True

            # Hunter Rank Progression
            ranks = [
                (1, "E-Rank Hunter"),
                (5, "D-Rank Hunter"),
                (10, "C-Rank Specialist"),
                (20, "B-Rank Commander"),
                (35, "A-Rank Sovereign"),
                (50, "S-Rank Shadow Monarch")
            ]
            current_rank = stats["rank"]
            for threshold_lvl, rank_title in ranks:
                if lvl >= threshold_lvl:
                    current_rank = rank_title

            if leveled_up:
                notifications.append(f"★ LEVEL UP! You reached Level {lvl:03d}!")
                notifications.append(f"Hunter Title Updated: {current_rank}")

            cursor.execute('''
                UPDATE hunter_profile
                SET level=?, exp=?, exp_next=?, gold=?, rank=?, tasks_solved=?
                WHERE id=1
            ''', (lvl, exp, exp_next, gold, current_rank, tasks))
            conn.commit()

        return notifications

    def get_active_quests(self) -> List[Dict[str, Any]]:
        """Retrieves active (incomplete) quests."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, quest_name, description, reward_exp, reward_gold FROM quests WHERE completed=0")
            rows = cursor.fetchall()
            return [
                {"id": r[0], "name": r[1], "desc": r[2], "exp": r[3], "gold": r[4]}
                for r in rows
            ]

    def complete_quest(self, quest_name: str) -> Optional[List[str]]:
        """Marks a quest complete and distributes rewards."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, reward_exp, reward_gold FROM quests WHERE quest_name=? AND completed=0", (quest_name,))
            row = cursor.fetchone()
            if not row:
                return None
            qid, exp_r, gold_r = row
            cursor.execute("UPDATE quests SET completed=1 WHERE id=?", (qid,))
            conn.commit()

        notifs = self.award_progress(exp_r, gold_r, f"Quest: {quest_name}")
        return notifs


# =====================================================================
# STANDALONE VERIFICATION & HUD PREVIEW
# =====================================================================

def main():
    print("=" * 70)
    print("  MIKU SYSTEM-LIFE OS BRIDGE (LitRPG Gamification & HUD Preview)")
    print("=" * 70)

    bridge = SystemLifeOSBridge()
    hud = SoloLevelingHUD()

    # Initial Stats Window
    stats = bridge.get_stats()
    print("\n" + hud.format_status_screen(stats) + "\n")

    # Simulate Miku Autonomous Accomplishments
    print("[Action] Miku completed automated Python calculation task...")
    notifs = bridge.award_progress(exp_gain=120, gold_gain=40, action_name="Clean REPL Execution")
    print(hud.system_window("SYSTEM REWARD NOTIFICATION", notifs))

    print("\n[Action] Miku triggered and solved active quest...")
    quest_notifs = bridge.complete_quest("First Sandbox Step")
    if quest_notifs:
        print(hud.system_window("QUEST CLEAR CONFIRMATION", quest_notifs))

    # Updated Stats Window
    updated_stats = bridge.get_stats()
    print("\n" + hud.format_status_screen(updated_stats) + "\n")

    print("✓ Phase II LitRPG System-Life OS Bridge and HUD fully operational.")


if __name__ == "__main__":
    main()
