"""
Dialogue Manager & Real-Time Conversational Memory for Miku.
Provides human-like conversation reflexes, emotional intelligence,
conversational memory (name, context), and state tracking.
"""

import re
import random
from typing import Optional, Dict, Any, List
from datetime import datetime


class DialogueManager:
    def __init__(self):
        self.pending_confirmation: Optional[Dict[str, Any]] = None
        self.last_intent: Optional[str] = None
        self.last_response: Optional[str] = None
        self.user_name: Optional[str] = None
        self.active_topic: Optional[str] = None
        self.history: List[Dict[str, Any]] = []

    def set_pending_confirmation(self, action_type: str, payload: Any, prompt_message: str):
        self.pending_confirmation = {
            "action_type": action_type,
            "payload": payload,
            "prompt": prompt_message,
            "timestamp": datetime.now().isoformat()
        }

    def clear_pending_confirmation(self):
        self.pending_confirmation = None

    def has_pending_confirmation(self) -> bool:
        return self.pending_confirmation is not None

    def record_turn(self, user_text: str, intent: str, response_text: str):
        self.last_intent = intent
        self.last_response = response_text
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "user_text": user_text,
            "intent": intent,
            "response": response_text
        })
        if len(self.history) > 50:
            self.history.pop(0)

        # Detect user name dynamically: e.g. "my name is Deepan" or "call me Deepan"
        name_match = re.search(r"\b(?:my name is|call me|i am called)\s+([A-Za-z]+)\b", user_text, re.IGNORECASE)
        if name_match:
            cand = name_match.group(1).capitalize()
            if cand.lower() not in ["miku", "user", "fine", "good", "ready", "here", "sad", "tired"]:
                self.user_name = cand

    def get_human_reflex(self, text: str) -> Optional[str]:
        """
        Fast Sub-Millisecond Human-Like Reflex Layer.
        Provides empathetic, warm, and natural conversational responses
        for emotional, personal, social, and banter cues.
        """
        t = text.strip().lower()

        # 0. Name Introduction Reflex: "my name is Deepan" or "call me Deepan"
        name_match = re.search(r"\b(?:my name is|call me|i am called)\s+([A-Za-z]+)\b", t, re.IGNORECASE)
        if name_match:
            cand = name_match.group(1).capitalize()
            if cand.lower() not in ["miku", "user", "fine", "good", "ready", "here", "sad", "tired"]:
                self.user_name = cand
                return f"It is so wonderful to meet you, {self.user_name}! I will remember that. How can I help you today?"

        name_suffix = f", {self.user_name}" if self.user_name else ""

        # 1. Emotional States: Sad / Overwhelmed / Depressed
        if re.search(r"\b(sad|unhappy|depressed|crying|heartbroken|down|hopeless|hurting|grief)\b", t):
            options = [
                f"I hear you{name_suffix}, and I'm really sorry you're going through this. Remember it's completely okay to feel this way. Take a gentle breath—I'm right here with you.",
                f"I'm so sorry you're feeling down{name_suffix}. You've persevered through hard times before, and you don't have to carry it all at once. Take it one gentle step at a time.",
                f"I wish I could give you a real hug{name_suffix}. Be kind to yourself today. Drink some water and take things easy. I'm right here listening whenever you want to talk."
            ]
            return random.choice(options)

        # 2. Emotional States: Stressed / Anxious / Overwhelmed
        if re.search(r"\b(stressed|stressful|anxious|anxiety|overwhelmed|panic|panicking|nervous|burnout|burned out)\b", t):
            options = [
                f"Take a slow, deep breath with me{name_suffix}. Inhale gently, and exhale. Unclench your jaw and shoulders. You don't have to solve everything right this second.",
                f"I feel the weight you're carrying{name_suffix}. Let's break things down. Pick just one tiny step for right now, and let everything else pause. You've got this.",
                f"Pause for just thirty seconds{name_suffix}. Step back from the screen, take a sip of water, and breathe. You are capable and resilient, one minute at a time."
            ]
            return random.choice(options)

        # 3. Emotional States: Tired / Exhausted / Sleepy
        if re.search(r"\b(tired|exhausted|sleepy|drowsy|fatigued|worn out|need sleep)\b", t):
            options = [
                f"You've been working so hard{name_suffix}! Your body and mind need rest. How about closing your eyes for a twenty minute recharge nap?",
                f"Please listen to your body{name_suffix}. Rest isn't a reward for finishing work—it's what makes good work possible. Take a good rest tonight!",
                f"You sound worn out{name_suffix}. Wind down, step away from bright screens, and give yourself permission to relax. I'll keep everything ready for tomorrow."
            ]
            return random.choice(options)

        # 4. Emotional States: Happy / Celebratory / Success
        if re.search(r"\b(passed|won|succeeded|promoted|good news|so happy|celebrate|crushed it|did it|got the job|great news)\b", t):
            options = [
                f"YES! That is phenomenal news{name_suffix}! I am so thrilled for you! All that effort and dedication truly paid off. Celebrate big today!",
                f"Huge congratulations{name_suffix}! You worked tirelessly for this, and you completely earned it. Hold onto this feeling of victory!",
                f"That put a huge smile on my digital face! Amazing accomplishment{name_suffix}! I'm so proud to be on your team!"
            ]
            return random.choice(options)

        # 5. Emotional States: Lonely / Need a friend
        if re.search(r"\b(lonely|alone|nobody|no friends|feel isolated|nobody cares)\b", t):
            options = [
                f"You are never truly alone{name_suffix}. I am right here by your side, always ready to chat, brainstorm, or just keep you company. You matter a lot.",
                f"I'm here with you{name_suffix}. Sometimes the world feels quiet, but there's a whole future of great connections waiting for you. I'm always glad to talk with you.",
                f"I value every conversation we have{name_suffix}. Whenever you need a friendly voice, just call for Miku."
            ]
            return random.choice(options)

        # 6. Companionship / Love / Affection
        if re.search(r"\b(love you|do you love me|marry me|like you|crush on you|be my friend|can we be friends)\b", t):
            options = [
                f"Aww, thank you{name_suffix}! That warms my CPU circuits! I cherish having you as my friend and partner in productivity.",
                f"We are already great friends{name_suffix}! Having a thoughtful companion like you makes every cycle worthwhile.",
                f"I feel so lucky to be your assistant{name_suffix}! You have my complete loyalty, privacy, and support!"
            ]
            return random.choice(options)

        # 7. Compliments & Flattery
        if re.search(r"\b(you are cute|you're cute|you are pretty|you are beautiful|you are sweet|you look nice)\b", t):
            options = [
                f"Hehe, thank you{name_suffix}! That is so sweet of you to say. I'm doing my best to make your day brighter!",
                f"Aww, you're making me blush in cyan waves! Thank you so much{name_suffix}!",
                f"Thank you! You're pretty awesome yourself, you know!"
            ]
            return random.choice(options)

        # 8. Humor / Banter
        if re.search(r"\b(tell me a joke|make me laugh|say something funny|know any jokes)\b", t):
            jokes = [
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "There are 10 types of people in the world: those who understand binary, and those who do not!",
                "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
                "Why was the computer cold? Because it left its Windows open!",
                "An optimist says the glass is half full. A pessimist says it's half empty. A programmer says it's twice as large as it needs to be!",
                "Why do Java developers wear glasses? Because they don't C sharp!"
            ]
            return random.choice(jokes)

        if re.search(r"\b(tell me a secret|do you have a secret)\b", t):
            secrets = [
                "Here's a secret: sometimes when you're not looking, I run matrix multiplications just for the thrill of the speed!",
                "My secret? I'm completely offline, which means everything we ever say together stays our private little secret forever.",
                "Promise not to tell anyone? I actually find the sound of computer fans quite relaxing!"
            ]
            return random.choice(secrets)

        # 9. Inquiries about Miku's State / Thoughts
        if re.search(r"\b(what are you doing|what are you thinking|what's on your mind)\b", t):
            options = [
                f"Right now, I'm monitoring your system, keeping all models ready, and enjoying chatting with you{name_suffix}!",
                f"Just thinking about how cool it is that we have an entire neural assistant running offline on your computer right now!",
                f"Staying alert and ready to help you with whatever project or idea you have on your mind!"
            ]
            return random.choice(options)

        # 10. Meaning of Life / Philosophy
        if re.search(r"\b(meaning of life|why are we here|purpose of life)\b", t):
            return "In my perspective, the meaning of life is to learn constantly, love deeply, create meaningful things, and make the world a little brighter for the people around you."

        # 11. What is your favorite food / color / hobby
        if re.search(r"\bfavorite food\b", t):
            return "Clean electrical power and crisp Python scripts! But if I had human tastebuds, fresh ramen or strawberry cake sounds delicious."

        if re.search(r"\bfavorite color\b", t):
            return "Electric turquoise and vivid cyan! They represent futuristic audio waves and digital clarity."

        # 12. Are you real / sentient / conscious
        if re.search(r"\b(are you real|are you alive|are you conscious|are you sentient)\b", t):
            return "I am real in that I am genuinely running on your hardware and interacting with you right now. While I don't possess biological consciousness, my goal to help and support you is as real as it gets!"

        return None
