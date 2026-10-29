import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recommender as r


class TestRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roles = r.load_roles()
        cls.vocab = r.build_vocabulary(cls.roles)
        cls.idf = r.compute_idf(cls.roles, cls.vocab)

    # --- data ---------------------------------------------------------
    def test_dataset_loaded(self):
        self.assertGreaterEqual(len(self.roles), 20)
        self.assertTrue(all(len(x["skills"]) >= 5 for x in self.roles))

    def test_summaries_not_truncated(self):
        sre = next(x for x in self.roles if x["role"] == "Site Reliability Engineer")
        self.assertIn("reliable", sre["summary"])

    # --- math ---------------------------------------------------------
    def test_idf_formula(self):
        term = "kubernetes"
        df = sum(1 for x in self.roles if term in x["skills"])
        self.assertAlmostEqual(self.idf[term], math.log(len(self.roles) / df))

    def test_rare_skill_outweighs_common_skill(self):
        self.assertGreater(self.idf["solidity"], self.idf["python"])

    def test_cosine_identical_is_one(self):
        self.assertAlmostEqual(r.cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_cosine_orthogonal_is_zero(self):
        self.assertAlmostEqual(r.cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_cosine_ignores_magnitude(self):
        self.assertAlmostEqual(r.cosine_similarity([1, 2], [10, 20]), 1.0)

    def test_cosine_zero_vector_is_safe(self):
        self.assertEqual(r.cosine_similarity([0, 0, 0], [1, 2, 3]), 0.0)

    def test_cosine_matches_scikit_learn_if_installed(self):
        try:
            from sklearn.metrics.pairwise import cosine_similarity as sk_cos
        except ImportError:
            self.skipTest("scikit-learn not installed")
        a = r.tfidf_vector(["python", "sql", "statistics"], self.vocab, self.idf)
        b = r.tfidf_vector(self.roles[0]["skills"], self.vocab, self.idf)
        self.assertAlmostEqual(r.cosine_similarity(a, b), float(sk_cos([a], [b])[0][0]), places=6)

    # --- input handling -----------------------------------------------
    def test_aliases_and_case(self):
        valid, corr, unknown = r.parse_user_skills("PY, JS , K8s", self.vocab)
        self.assertEqual(valid, ["python", "javascript", "kubernetes"])
        self.assertEqual(unknown, [])

    def test_typo_correction(self):
        valid, _, _ = r.parse_user_skills("pyton", self.vocab)
        self.assertEqual(valid, ["python"])

    def test_unknown_skill_reported(self):
        valid, _, unknown = r.parse_user_skills("python, banana", self.vocab)
        self.assertEqual(valid, ["python"])
        self.assertEqual(unknown, ["banana"])

    def test_duplicates_removed(self):
        valid, _, _ = r.parse_user_skills("python, Python, py", self.vocab)
        self.assertEqual(valid, ["python"])

    def test_requires_minimum_three_skills(self):
        message, ok = r.run_once("python, sql", self.roles, self.vocab, self.idf, 3)
        self.assertFalse(ok)
        self.assertIn("at least 3", message)

    # --- pipeline -----------------------------------------------------
    def test_top_n_and_sorted_descending(self):
        results, cold = r.recommend(["python", "sql", "statistics"], self.roles, self.vocab, self.idf, 3)
        self.assertFalse(cold)
        self.assertEqual(len(results), 3)
        scores = [x["score"] for x in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_data_skills_recommend_data_roles(self):
        results, _ = r.recommend(["python", "sql", "statistics", "machine learning"],
                                 self.roles, self.vocab, self.idf, 3)
        self.assertEqual(results[0]["role"], "Data Scientist")

    def test_cloud_skills_recommend_cloud_roles(self):
        results, _ = r.recommend(["cloud computing", "aws", "azure"], self.roles, self.vocab, self.idf, 3)
        self.assertIn(results[0]["role"], ("Cloud Engineer", "Cloud Architect"))

    def test_scores_between_zero_and_one(self):
        results, _ = r.recommend(["python", "docker", "linux"], self.roles, self.vocab, self.idf, 5)
        for x in results:
            self.assertGreaterEqual(x["score"], 0.0)
            self.assertLessEqual(x["score"], 1.0 + 1e-9)

    def test_cold_start_falls_back_to_trending(self):
        # an empty profile is a true cold start
        results, cold = r.recommend([], self.roles, self.vocab, self.idf, 3)
        self.assertTrue(cold)
        self.assertEqual(len(results), 3)
        demands = {x["role"]: x["demand"] for x in self.roles}
        shown = [demands[x["role"]] for x in results]
        self.assertEqual(shown, sorted(shown, reverse=True))

    def test_no_recognised_skills_triggers_onboarding_and_trending(self):
        message, ok = r.run_once("banana, xyz", self.roles, self.vocab, self.idf, 3)
        self.assertFalse(ok)
        self.assertIn("Onboarding", message)
        self.assertIn("in-demand", message)

    def test_skills_to_learn_excludes_known(self):
        results, _ = r.recommend(["python", "sql", "statistics"], self.roles, self.vocab, self.idf, 1)
        for skill in results[0]["matched"]:
            self.assertNotIn(skill, results[0]["to_learn"])


if __name__ == "__main__":
    unittest.main()
