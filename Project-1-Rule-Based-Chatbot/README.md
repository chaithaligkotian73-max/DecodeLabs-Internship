# 🤖 Rule-Based AI Chatbot

**DecodeLabs AI Internship – Project 1 (Batch 2026)**

A simple rule-based chatbot built with pure Python control flow and a dictionary-based knowledge base. No external libraries required.

## ✨ Features
- Continuous `while True` input loop
- Input sanitization (case, whitespace, trailing punctuation)
- Knowledge base with 15+ intents (dictionary / hash map, O(1) lookup)
- Dynamic replies for `time` and `date`
- Random responses for variety (e.g. jokes)
- Fallback response for unknown input using `dict.get()`
- Clean exit via `bye`, `exit`, `quit`, `goodbye`, `see you` (also Ctrl+C)

## 🧱 Architecture (IPO Model)
| Stage   | What happens                                   |
|---------|------------------------------------------------|
| Input   | `input()` → `sanitize()` (lower, strip, clean) |
| Process | Dictionary lookup in `RESPONSES`               |
| Output  | Print reply, loop back                         |

## 🚀 How to Run
1. Install Python 3.8+ from https://python.org
2. Open this folder in VS Code (`File > Open Folder`)
3. Open `chatbot.py` and press **F5** (or the ▶ Run button), or in the terminal:

```bash
python chatbot.py
```

## 💬 Example
```
DecodeBot: Hello! I'm DecodeBot. Type 'help' for options or 'bye' to exit.
You:   HELLO!!
DecodeBot: Hi there!
You: what is ai
DecodeBot: Artificial Intelligence is the ability of machines ...
You: blah
DecodeBot: I do not understand. Type 'help' to see what I can do.
You: bye
DecodeBot: Goodbye! Have a great day.
```

## 🧪 Run Tests
```bash
python -m unittest discover tests
```

## ➕ Add Your Own Intent
Edit the `RESPONSES` dictionary in `chatbot.py`:
```python
"favorite color": "Blue, like a blueprint!",
```

## 📤 Upload to GitHub
```bash
git init
git add .
git commit -m "Project 1: Rule-based AI chatbot"
git branch -M main
git remote add origin https://github.com/<your-username>/rule-based-chatbot.git
git push -u origin main
```

## 🔮 Next Step (Project 2)
Move from exact-match keys to semantic matching with embeddings.
