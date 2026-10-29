# 🎯 Tech Stack Recommender (AI Recommendation Logic)

**DecodeLabs AI Internship – Project 3 (Batch 2026)**

A content-based recommendation engine, written in pure Python with **no external libraries**. You enter your skills, and it recommends the Top 3 career paths that best align with them using **TF-IDF weighting** and **cosine similarity**.

## ✨ How it works (IPO model)

| Stage | What happens |
|-------|--------------|
| **Input** | The user types 3 or more skills, e.g. `python, cloud computing, automation` |
| **Process** | Skills and job roles become TF-IDF vectors in one shared vocabulary → cosine similarity |
| **Output** | Top-N ranked roles with match %, matched skills and skills to learn next |

### The 4-step ranking pipeline
1. **Ingestion**: clean the input (case, spaces, aliases like `js` → `javascript`, typo fixes like `pyton` → `python`)
2. **Scoring**: cosine similarity between the user vector and every role vector
3. **Sorting**: highest score first (demand breaks ties)
4. **Filtering**: keep only the Top N to avoid choice overload

### The math
```
TF  = (count of term in document) / (total terms in document)
IDF = log( total documents / documents containing the term )
cos(θ) = (A · B) / (‖A‖ ‖B‖)
```
Rare, specific skills (e.g. `kubernetes`) get a higher weight than skills almost every role shares (e.g. `git`). Cosine similarity ignores vector magnitude, so a long skill list doesn't unfairly beat a short one.

### Cold start handling
- **User cold start**: if no skills are recognised, the tool shows an onboarding tip and falls back to **trending (most in-demand) roles**. Type `skip` to see them directly.
- **Item cold start**: a new job role only needs one new row in `data/raw_skills.csv` and it is recommended immediately.

## 🚀 How to run

No installation needed, just Python 3.8+.

```bash
python recommender.py
```

Other ways to run it:
```bash
python recommender.py --skills "python, cloud computing, automation"
python recommender.py --skills "python, sql, statistics" --top 5
```

Inside the program: `list` shows all known skills, `skip` shows trending roles, `q` quits.

Run the tests:
```bash
python -m unittest discover tests
```

## 💬 Example
```
Your skills: python, sql, statistics, machine learning

Top 3 career paths for you:

1. Data Scientist
   Match    :  53.6%  [###########         ]
   You have : python, sql, machine learning, statistics
   Learn next: data analysis, pandas, numpy, data visualization, scikit-learn
   About    : Turns raw data into predictions and business insights.
```

## 📁 Project structure
```
Project-3-Recommendation-Logic/
├── recommender.py        # main program (pipeline + interactive loop)
├── data/raw_skills.csv   # 23 job roles = the "items"
├── tests/                # 21 unit tests
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## ➕ Add your own role
Add a row to `data/raw_skills.csv`:
```
role,skills,demand,summary
Robotics Engineer,python;c++;embedded systems;computer vision;algorithms;linux,70,Builds software for robots.
```
Skills must come from the existing skill list (type `list` to see it), or add the same skill to other roles so it joins the shared vocabulary.

## 🔮 Next step
Move from content-based filtering to collaborative filtering and neural recommenders.
