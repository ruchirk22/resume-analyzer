    # tests/test_matcher.py
import sys
import os
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.matcher import calculate_cosine_similarity, categorize_candidate

def test_cosine_similarity():
        """
        Tests the cosine similarity calculation.
        """
        print("\n--- Testing Cosine Similarity ---")
        vec_a = np.array([1, 1, 1, 1])
        vec_b = np.array([1, 1, 1, 1])
        vec_c = np.array([-1, -1, -1, -1])
        vec_d = np.array([1, 0, 0, 0])

        # A vector with itself should be perfectly similar (1.0)
        similarity_self = calculate_cosine_similarity(vec_a, vec_b)
        assert np.isclose(similarity_self, 1.0), f"Expected 1.0, got {similarity_self}"
        print(f"Similarity with self: {similarity_self:.2f} (Passed)")

        # A vector with its opposite should have a low similarity
        # For non-negative vectors, this will be 0 after clipping
        similarity_opposite = calculate_cosine_similarity(vec_a, vec_c)
        assert np.isclose(similarity_opposite, 0.0), f"Expected 0.0 after clipping, got {similarity_opposite}"
        print(f"Similarity with opposite vector: {similarity_opposite:.2f} (Passed)")

        # Orthogonal vectors should have 0 similarity
        # Here we test a partially overlapping vector
        similarity_partial = calculate_cosine_similarity(vec_a, vec_d)
        assert np.isclose(similarity_partial, 0.5), f"Expected 0.5, got {similarity_partial}"
        print(f"Similarity with partial vector: {similarity_partial:.2f} (Passed)")

def test_categorization():
        """
        Tests the candidate categorization based on similarity scores.
        """
        print("\n--- Testing Candidate Categorization ---")
        # Test "Strong" category
        strong_result = categorize_candidate(0.80)
        assert strong_result["category"] == "Strong" and strong_result["color"] == "green"
        print(f"Score 0.80 -> {strong_result['category']} (Passed)")

        # Test "Good" category
        good_result = categorize_candidate(0.70)
        assert good_result["category"] == "Good" and good_result["color"] == "yellow"
        print(f"Score 0.70 -> {good_result['category']} (Passed)")

        # Test "Average" category
        average_result = categorize_candidate(0.60)
        assert average_result["category"] == "Average" and average_result["color"] == "orange"
        print(f"Score 0.60 -> {average_result['category']} (Passed)")

        # Test "Weak" category
        weak_result = categorize_candidate(0.50)
        assert weak_result["category"] == "Weak" and weak_result["color"] == "red"
        print(f"Score 0.50 -> {weak_result['category']} (Passed)")


if __name__ == "__main__":
        test_cosine_similarity()
        test_categorization()
        print("\nAll matcher tests passed!")
    
