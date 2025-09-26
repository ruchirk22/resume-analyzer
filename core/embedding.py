# core/embedding.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

# --- Phase 1: Using a local Sentence Transformer model ---
# This is a good starting point. For Phase 3/4, this could be replaced
# with a call to a more powerful API-based embedding model if needed.
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
model = None

def load_embedding_model() -> SentenceTransformer:
    """
    Loads the sentence-transformer model from Hugging Face.
    We load it once and reuse it to save memory and time.
    """
    global model
    if model is None:
        print("Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded.")
    return model

def generate_embedding(sentences: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Generates a single embedding for a list of sentences by averaging them.

    Args:
        sentences (List[str]): A list of sentences from a resume or JD.
        model (SentenceTransformer): The loaded sentence transformer model.

    Returns:
        np.ndarray: A single numpy array representing the average embedding of the text.
    """
    if not sentences:
        # The model 'all-MiniLM-L6-v2' has an embedding dimension of 384
        return np.zeros(384)

    # Generate embeddings for all sentences
    embeddings = model.encode(sentences, show_progress_bar=False)

    # Calculate the mean of the embeddings to get a single representative vector
    mean_embedding = np.mean(embeddings, axis=0)

    return mean_embedding
