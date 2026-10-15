"""
Rule-Based AI Chatbot  |  DecodeLabs AI Internship - Project 1
================================================================
Architecture (IPO model):
    INPUT   -> sanitize raw text (lowercase, strip whitespace/punctuation)
    PROCESS -> dictionary lookup (O(1)) for intent matching
    OUTPUT  -> print the reply, then loop back

Features:
    * Continuous `while True` loop
    * Input sanitization (case, whitespace, trailing punctuation)
    * Knowledge base (dictionary) with 5+ intents
    * Fallback response for unknown inputs via dict.get()
    * Clean exit with `break`
"""

import random
import string
from datetime import datetime

BOT_NAME = "DecodeBot"

# ---------------------------------------------------------------------------
# KNOWLEDGE BASE  (key -> value pairs, O(1) lookup)
# Values can be a string or a list of strings (a random one is chosen).
# ---------------------------------------------------------------------------
RESPONSES = {
    # Greetings
    "hello": ["Hi there!", "Hello! How can I help you today?"],
    "hi": ["Hey! Nice to see you.", "Hi there!"],
    "hey": "Hey! What's up?",
    "good morning": "Good morning! Hope you have a great day.",
    "good afternoon": "Good afternoon! How can I assist you?",
    "good evening": "Good evening! How was your day?",
    # Small talk
    "how are you": "I'm just code, but I'm running perfectly. How about you?",
    "i am fine": "Glad to hear that!",
    "i'm fine": "Glad to hear that!",
    "what is your name": f"I'm {BOT_NAME}, a rule-based chatbot built at DecodeLabs.",
    "who are you": f"I'm {BOT_NAME}, a rule-based chatbot built at DecodeLabs.",
    "who made you": "I was built by an AI intern at DecodeLabs as Project 1.",
    "thanks": "You're welcome!",
    "thank you": "Happy to help!",
    # Topic intents
    "help": "Try: hello, how are you, what is ai, tell me a joke, time, date, bye.",
    "what is ai": "Artificial Intelligence is the ability of machines to perform tasks that normally need human intelligence.",
    "what is a chatbot": "A chatbot is a program that simulates human conversation.",
    "what is rule based": "A rule-based system follows explicit if-else / lookup rules. No learning, fully predictable.",
    "tell me a joke": [
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "There are 10 kinds of people: those who understand binary and those who don't.",
    ],
}

# Inputs that terminate the program (the "Kill Command")
EXIT_COMMANDS = {"bye", "exit", "quit", "goodbye", "see you"}

FALLBACK = "I do not understand. Type 'help' to see what I can do."


# ---------------------------------------------------------------------------
# PHASE 1: INPUT & SANITIZATION
# ---------------------------------------------------------------------------
def sanitize(raw_input: str) -> str:
    """Normalize text: lowercase, trim, drop trailing punctuation, collapse spaces."""
    clean = raw_input.lower().strip()
    clean = clean.rstrip(string.punctuation.replace("'", ""))  # keep apostrophes (i'm)
    return " ".join(clean.split())


# ---------------------------------------------------------------------------
# PHASE 2: PROCESS  (dynamic intents + dictionary lookup)
# ---------------------------------------------------------------------------
def get_response(clean_input: str) -> str:
    """Return the bot's reply for an already-sanitized input."""
    if clean_input in ("time", "what time is it"):
        return "The time is " + datetime.now().strftime("%I:%M %p") + "."
    if clean_input in ("date", "what is the date", "today"):
        return "Today is " + datetime.now().strftime("%A, %d %B %Y") + "."

    reply = RESPONSES.get(clean_input, FALLBACK)  # lookup + fallback in one step
    if isinstance(reply, list):
        reply = random.choice(reply)
    return reply


# ---------------------------------------------------------------------------
# PHASE 3: OUTPUT + THE HEARTBEAT (infinite loop)
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"{BOT_NAME}: Hello! I'm {BOT_NAME}. Type 'help' for options or 'bye' to exit.")

    while True:
        try:
            raw_input_text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{BOT_NAME}: Goodbye!")
            break

        clean_input = sanitize(raw_input_text)

        if clean_input == "":
            print(f"{BOT_NAME}: Please type something.")
            continue

        if clean_input in EXIT_COMMANDS:
            print(f"{BOT_NAME}: Goodbye! Have a great day.")
            break

        print(f"{BOT_NAME}: {get_response(clean_input)}")


if __name__ == "__main__":
    main()
