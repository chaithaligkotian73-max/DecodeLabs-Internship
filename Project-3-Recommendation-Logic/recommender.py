"""
Tech Stack Recommender (AI Recommendation Logic)
DecodeLabs AI Internship - Project 3
================================================
A content-based recommendation engine written in pure Python (no external
libraries). Job roles are the "items"; the user's skills are the "profile".

IPO model:
    INPUT   -> the user types 3+ skills
    PROCESS -> skills -> TF-IDF vectors (shared vocabulary) -> cosine similarity
    OUTPUT  -> Top-N most relevant career paths

4-step ranking pipeline:
    1. Ingestion  2. Scoring  3. Sorting  4. Filtering (Top-N)

Cold start handling:
    * User cold start -> onboarding prompt + trending (demand-based) fallback
    * Item cold start -> a new role only needs a row in data/raw_skills.csv

Run:
    python recommender.py
    python recommender.py --skills "python, cloud computing, automation"
    python recommender.py --skills "python, sql, statistics" --top 5
"""

import argparse
import csv
import difflib
import math
import os

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw_skills.csv")
MIN_SKILLS = 3
DEFAULT_TOP_N = 3
EXIT_WORDS = {"q", "quit", "exit", "bye"}

# Vocabulary alignment: different spellings must map to the SAME canonical term,
# otherwise the similarity math fails ("js" vs "javascript").
ALIASES = {
    "py": "python", "python3": "python",
    "js": "javascript", "ts": "typescript",
    "ml": "machine learning", "machine-learning": "machine learning",
    "dl": "deep learning", "deep-learning": "deep learning",
    "ai": "machine learning",
    "cloud": "cloud computing", "cloud-computing": "cloud computing",
    "k8s": "kubernetes", "kube": "kubernetes",
    "postgres": "postgresql", "psql": "postgresql", "mongo": "mongodb",
    "node": "node.js", "nodejs": "node.js", "node js": "node.js",
    "reactjs": "react", "react.js": "react",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "tf": "tensorflow", "torch": "pytorch",
    "cicd": "ci/cd", "ci cd": "ci/cd", "ci-cd": "ci/cd", "pipelines": "ci/cd",
    "viz": "data visualization", "data viz": "data visualization",
    "powerbi": "power bi", "power-bi": "power bi",
    "springboot": "spring boot", "spring": "spring boot",
    "cpp": "c++", "csharp": "c#",
    "amazon web services": "aws", "google cloud": "gcp",
    "natural language processing": "nlp",
    "cv": "computer vision", "ux": "ux research",
    "api": "apis", "rest api": "rest", "restful": "rest",
    "unit testing": "testing", "test": "testing", "tests": "testing",
    "shell": "bash", "shell scripting": "bash", "script": "scripting",
    "infrastructure as code": "terraform", "iac": "terraform",
    "ds": "data structures", "dsa": "algorithms",
    "network": "networking", "cyber security": "security", "cybersecurity": "security",
    "data engineering": "etl", "big data": "spark",
}


# ---------------------------------------------------------------------------
# DATA LOADING  (the "items")
# ---------------------------------------------------------------------------
def load_roles(path=DATA_FILE):
    """Read raw_skills.csv -> list of role dicts."""
    roles = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            skills = [s.strip().lower() for s in row["skills"].split(";") if s.strip()]
            roles.append({
                "role": row["role"].strip(),
                "skills": skills,
                "demand": int(row.get("demand") or 0),
                "summary": (row.get("summary") or "").strip(),
            })
    return roles


def build_vocabulary(roles):
    """The shared vocabulary space: every unique skill in the dataset, sorted."""
    return sorted({skill for role in roles for skill in role["skills"]})


# ---------------------------------------------------------------------------
# VECTOR MAPPING + TF-IDF   (slide formulas)
#   TF  = count of term t in document d / total terms in d
#   IDF = log( total documents / documents containing t )
# ---------------------------------------------------------------------------
def compute_idf(roles, vocabulary):
    n_docs = len(roles)
    idf = {}
    for term in vocabulary:
        doc_freq = sum(1 for role in roles if term in role["skills"])
        idf[term] = math.log(n_docs / doc_freq)
    return idf


def tfidf_vector(terms, vocabulary, idf):
    """Turn a list of terms into a TF-IDF weighted vector over the vocabulary."""
    total = len(terms)
    vector = []
    for term in vocabulary:
        tf = terms.count(term) / total if total else 0.0
        vector.append(tf * idf[term])
    return vector


# ---------------------------------------------------------------------------
# SIMILARITY ENGINE   cos(theta) = (A . B) / (||A|| * ||B||)
# ---------------------------------------------------------------------------
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:      # zero vector = cold start, avoid divide-by-zero
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# STEP 1: INGESTION  (clean + align user input with the vocabulary)
# ---------------------------------------------------------------------------
def normalize_skill(raw, vocabulary):
    """
    Return (canonical_skill or None, was_corrected).
    Handles case, spacing, aliases ("js" -> "javascript") and small typos.
    """
    text = " ".join(raw.lower().strip().strip(".,;:!?").split())
    if not text:
        return None, False
    if text in vocabulary:
        return text, False
    if text in ALIASES and ALIASES[text] in vocabulary:
        return ALIASES[text], True
    close = difflib.get_close_matches(text, vocabulary, n=1, cutoff=0.85)
    if close:
        return close[0], True
    return None, False


def parse_user_skills(text, vocabulary):
    """
    Split a comma-separated line into (valid, corrections, unknown).
    valid       : unique canonical skills, order preserved
    corrections : [(what_user_typed, what_we_understood)]
    unknown     : inputs we could not match
    """
    valid, corrections, unknown = [], [], []
    for raw in text.replace(";", ",").split(","):
        if not raw.strip():
            continue
        skill, corrected = normalize_skill(raw, vocabulary)
        if skill is None:
            unknown.append(raw.strip())
        else:
            if corrected:
                corrections.append((raw.strip(), skill))
            if skill not in valid:
                valid.append(skill)
    return valid, corrections, unknown


# ---------------------------------------------------------------------------
# STEPS 2-4: SCORING -> SORTING -> FILTERING
# ---------------------------------------------------------------------------
def trending_fallback(roles, top_n=DEFAULT_TOP_N):
    """Cold-start bypass: recommend globally in-demand roles."""
    ranked = sorted(roles, key=lambda r: r["demand"], reverse=True)[:top_n]
    return [{
        "role": r["role"], "score": 0.0, "matched": [],
        "to_learn": r["skills"][:5], "summary": r["summary"],
    } for r in ranked]


def recommend(user_skills, roles, vocabulary, idf, top_n=DEFAULT_TOP_N):
    """
    Returns (results, cold_start_flag).
    Each result: {role, score, matched, to_learn, summary}
    """
    user_vec = tfidf_vector(user_skills, vocabulary, idf)

    # 2. SCORING
    scored = []
    for role in roles:
        role_vec = tfidf_vector(role["skills"], vocabulary, idf)
        score = cosine_similarity(user_vec, role_vec)
        scored.append({
            "role": role["role"],
            "score": score,
            "matched": [s for s in role["skills"] if s in user_skills],
            "to_learn": [s for s in role["skills"] if s not in user_skills],
            "summary": role["summary"],
            "demand": role["demand"],
        })

    # cold start: user vector is all zeros, or nothing matches any role
    if all(item["score"] == 0 for item in scored):
        return trending_fallback(roles, top_n), True

    # 3. SORTING (highest score first; demand breaks ties)
    scored.sort(key=lambda item: (item["score"], item["demand"]), reverse=True)

    # 4. FILTERING (Top-N only, to prevent choice overload)
    return scored[:top_n], False


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
def format_results(results, cold_start=False):
    lines = []
    if cold_start:
        lines.append("Cold start: not enough personal data to match yet.")
        lines.append("Showing the most in-demand career paths for now:\n")
    else:
        lines.append(f"Top {len(results)} career paths for you:\n")
    for rank, item in enumerate(results, start=1):
        bar = "#" * round(item["score"] * 20)
        lines.append(f"{rank}. {item['role']}")
        if not cold_start:
            lines.append(f"   Match    : {item['score'] * 100:5.1f}%  [{bar:<20}]")
            lines.append(f"   You have : {', '.join(item['matched']) or '-'}")
        lines.append(f"   Learn next: {', '.join(item['to_learn'][:5]) or '-'}")
        lines.append(f"   About    : {item['summary']}")
        lines.append("")
    return "\n".join(lines)


def run_once(text, roles, vocabulary, idf, top_n):
    """Parse one line of user input and return the text to display."""
    valid, corrections, unknown = parse_user_skills(text, vocabulary)
    out = []
    for typed, understood in corrections:
        out.append(f"  (understood '{typed}' as '{understood}')")
    if unknown:
        out.append(f"  (skipped, not in our skill list: {', '.join(unknown)})")
    if len(valid) == 0:
        # USER COLD START: empty profile -> onboarding hint + trending fallback
        starters = ["python", "sql", "git", "linux", "javascript", "docker"]
        out.append("We could not recognise any skills yet (new user / empty profile).")
        out.append("Onboarding tip - pick 3 you know, for example: " + ", ".join(starters[:6]))
        out.append("Meanwhile, here are the most in-demand paths:\n")
        out.append(format_results(trending_fallback(roles, top_n), cold_start=True))
        return "\n".join(out), False
    if len(valid) < MIN_SKILLS:
        out.append(f"Please give at least {MIN_SKILLS} valid skills "
                   f"(got {len(valid)}). Type 'list' to see all skills.")
        return "\n".join(out), False
    results, cold = recommend(valid, roles, vocabulary, idf, top_n)
    out.append(format_results(results, cold))
    return "\n".join(out), True


# ---------------------------------------------------------------------------
# INTERACTIVE LOOP
# ---------------------------------------------------------------------------
def interactive(roles, vocabulary, idf, top_n):
    print("=" * 62)
    print("  TECH STACK RECOMMENDER  |  DecodeLabs AI - Project 3")
    print("=" * 62)
    print(f"Enter at least {MIN_SKILLS} skills separated by commas.")
    print("Example: python, cloud computing, automation")
    print("Commands: 'list' = show all skills, 'skip' = see trending roles, 'q' = quit\n")

    while True:
        try:
            text = input("Your skills: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if text.lower() in EXIT_WORDS:
            print("Goodbye! Keep building.")
            break
        if text.lower() == "skip":
            print(format_results(trending_fallback(roles, top_n), cold_start=True))
            continue
        if text.lower() == "list":
            print("Available skills:\n  " + ", ".join(vocabulary) + "\n")
            continue
        if not text:
            print("Please type some skills (or 'list' for ideas).\n")
            continue
        message, _ = run_once(text, roles, vocabulary, idf, top_n)
        print(message)


def main():
    parser = argparse.ArgumentParser(description="Tech Stack Recommender")
    parser.add_argument("--skills", help='comma-separated skills, e.g. "python, sql, statistics"')
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="how many roles to show")
    parser.add_argument("--data", default=DATA_FILE, help="path to the roles CSV")
    args = parser.parse_args()

    roles = load_roles(args.data)
    vocabulary = build_vocabulary(roles)
    idf = compute_idf(roles, vocabulary)

    if args.skills:
        message, _ = run_once(args.skills, roles, vocabulary, idf, args.top)
        print(message)
    else:
        interactive(roles, vocabulary, idf, args.top)


if __name__ == "__main__":
    main()
