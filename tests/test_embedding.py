# tests/test_embedding.py
import sys
import os
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.embedding import load_embedding_model, generate_embedding

def test_embedding_generation():
    """
    Tests the embedding model loading and vector generation.
    """
    print("\n--- Testing Embedding Generation ---")
    # 1. Load the model
    model = load_embedding_model()
    assert model is not None, "Model should be loaded"
    
    # 2. Test with sample sentences
    sample_sentences = [
        "Experienced in Python and object-oriented programming.",
        "Developed machine learning models using TensorFlow and PyTorch.",
        "A collaborative team player with strong communication skills."
    ]
    
    embedding = generate_embedding(sample_sentences, model)
    
    # 3. Check the output
    assert isinstance(embedding, np.ndarray), "Embedding should be a numpy array"
    print(f"Embedding type: {type(embedding)}")
    
    # The 'all-MiniLM-L6-v2' model produces embeddings of dimension 384
    expected_shape = (384,)
    assert embedding.shape == expected_shape, f"Embedding shape should be {expected_shape}, but got {embedding.shape}"
    print(f"Embedding shape: {embedding.shape}")
    
    # 4. Test with empty input
    empty_embedding = generate_embedding([], model)
    assert empty_embedding.shape == expected_shape, "Empty input should return a zero vector of the correct shape"
    assert np.all(empty_embedding == 0), "Empty input should return a zero vector"
    print("Empty list test passed.")
    
    print("\nEmbedding generation test passed!")
    


if __name__ == "__main__":
    test_embedding_generation()
    
