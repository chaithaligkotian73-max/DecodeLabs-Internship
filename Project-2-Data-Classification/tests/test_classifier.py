import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import classifier as c


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.features, cls.classes = c.load_data()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = c.split_data(cls.X, cls.y)
        cls.Xs_train, cls.Xs_test, cls.scaler = c.scale_features(cls.X_train, cls.X_test)

    def test_dataset_shape(self):
        self.assertEqual(self.X.shape, (150, 4))
        self.assertEqual(len(self.classes), 3)

    def test_split_sizes(self):
        self.assertEqual(len(self.X_train), 120)
        self.assertEqual(len(self.X_test), 30)

    def test_split_is_stratified(self):
        self.assertTrue(np.array_equal(np.bincount(self.y_test), [10, 10, 10]))

    def test_scaling_uses_training_statistics(self):
        self.assertTrue(np.allclose(self.Xs_train.mean(axis=0), 0, atol=1e-8))
        self.assertTrue(np.allclose(self.Xs_train.std(axis=0), 1, atol=1e-8))

    def test_model_accuracy_is_high(self):
        model = c.train_knn(self.Xs_train, self.y_train, k=5)
        results = c.evaluate(model, self.Xs_test, self.y_test, self.classes)
        self.assertGreaterEqual(results["accuracy"], 0.90)
        self.assertEqual(results["confusion_matrix"].shape, (3, 3))

    def test_best_k_in_range(self):
        best_k, errors = c.find_best_k(self.Xs_train, self.y_train)
        self.assertIn(best_k, errors)
        self.assertEqual(len(errors), 25)

    def test_predict_flower(self):
        model = c.train_knn(self.Xs_train, self.y_train, k=5)
        self.assertEqual(c.predict_flower(model, self.scaler, self.classes, [5.1, 3.5, 1.4, 0.2]), "setosa")
        self.assertEqual(c.predict_flower(model, self.scaler, self.classes, [6.9, 3.1, 5.8, 2.1]), "virginica")


if __name__ == "__main__":
    unittest.main()
