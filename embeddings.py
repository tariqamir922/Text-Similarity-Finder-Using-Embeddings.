"""
embeddings.py
==============
Handles loading the Sentence Transformers model and converting text into
numerical embedding vectors.

Beginner Overview:
------------------
A text embedding is a list of numbers (a vector) that mathematically represents
the semantic meaning of a word, sentence, or document. Words and sentences that
share similar meanings will have vectors positioned close to each other in this
high-dimensional "vector space".

The model used here is 'all-MiniLM-L6-v2':
- It is lightweight (~80 MB) and runs fast even on standard CPUs.
- It transforms any text into a 384-dimensional dense vector.
- It is pre-trained on over 1 billion sentence pairs to understand English semantics.
"""

from typing import List, Optional
import numpy as np

# Global cache so the model is loaded into memory only once
_MODEL_CACHE: Optional[object] = None
DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"


def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME, force_reload: bool = False
):
    """
    Loads and caches the SentenceTransformer model.

    Loading a transformer model from disk/HuggingFace takes a few seconds and consumes
    memory. By caching the loaded model in a global variable, subsequent calls
    reuse the existing instance instead of reloading it on every comparison.

    Parameters:
        model_name (str): The identifier of the pre-trained Hugging Face model.
                          Defaults to 'all-MiniLM-L6-v2'.
        force_reload (bool): If True, reloads the model even if one is cached.

    Returns:
        SentenceTransformer: The ready-to-use embedding model instance.

    Raises:
        RuntimeError: If downloading or initializing the model fails.
        ImportError: If the sentence-transformers package is missing.
    """
    global _MODEL_CACHE

    if _MODEL_CACHE is not None and not force_reload:
        return _MODEL_CACHE

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The 'sentence-transformers' library is required but not installed.\n"
            "Please install it using: pip install sentence-transformers"
        ) from exc

    try:
        # Load the pre-trained model. If running for the first time, this downloads
        # the weights from Hugging Face Hub and saves them to local cache (~/.cache).
        model = SentenceTransformer(model_name)
        _MODEL_CACHE = model
        return model
    except Exception as exc:
        # Catch network or file errors and provide clear diagnostic advice
        raise RuntimeError(
            f"Failed to load sentence transformer model '{model_name}'.\n"
            f"Reason: {str(exc)}\n\n"
            "Troubleshooting Tips:\n"
            "1. Check your internet connection if this is your first time running the app.\n"
            "2. Ensure you have sufficient disk space in your user cache directory.\n"
            "3. If behind a corporate proxy, configure your HTTP_PROXY/HTTPS_PROXY variables."
        ) from exc


def generate_embedding(
    text: str, model=None
) -> np.ndarray:
    """
    Converts a single input string into a 1-dimensional numerical embedding vector.

    Step-by-step process:
    1. Validate that the input text is a non-empty string.
    2. Tokenize the input text into subword tokens understood by the model.
    3. Pass tokens through the Transformer layers (MiniLM) to capture contextual semantics.
    4. Apply mean pooling across token representations to produce a single sentence vector.
    5. Return the resulting vector as a 1D NumPy array (length 384 for all-MiniLM-L6-v2).

    Parameters:
        text (str): The sentence or paragraph to encode.
        model (SentenceTransformer, optional): A pre-loaded model instance. If None,
                                               loads the cached default model.

    Returns:
        np.ndarray: A 1D float32 array representing the text's semantic embedding.

    Raises:
        ValueError: If text is empty, contains only whitespace, or is not a string.
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected a string input, but received type '{type(text).__name__}'.")

    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Input text cannot be empty or contain only whitespace.")

    if model is None:
        model = load_embedding_model()

    # encode() produces a NumPy array representing the text in 384 dimensions.
    # convert_to_numpy=True guarantees standard NumPy float array output.
    embedding = model.encode(cleaned_text, convert_to_numpy=True)

    # Ensure the returned vector is 1D
    return np.asarray(embedding, dtype=np.float32).flatten()


def generate_embeddings(
    texts: List[str], model=None
) -> np.ndarray:
    """
    Converts a list of text strings into a 2D matrix of embeddings (N, dimension).

    Batch encoding is more efficient than encoding sentences one by one.

    Parameters:
        texts (List[str]): List of sentences or paragraphs to encode.
        model (SentenceTransformer, optional): A pre-loaded model instance.

    Returns:
        np.ndarray: A 2D float32 array with shape (len(texts), embedding_dim).

    Raises:
        ValueError: If texts list is empty or any item is blank.
    """
    if not texts:
        raise ValueError("The 'texts' list cannot be empty.")

    cleaned_texts = []
    for idx, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Item at index {idx} is empty or not a valid string.")
        cleaned_texts.append(text.strip())

    if model is None:
        model = load_embedding_model()

    embeddings = model.encode(cleaned_texts, convert_to_numpy=True)
    return np.asarray(embeddings, dtype=np.float32)


def get_embedding_dimension(model=None) -> int:
    """
    Returns the dimensionality of the vector space used by the model.
    For 'all-MiniLM-L6-v2', this is 384.

    Parameters:
        model (SentenceTransformer, optional): The model instance.

    Returns:
        int: Vector dimension size.
    """
    if model is None:
        model = load_embedding_model()

    # Check for newer API get_embedding_dimension(), then legacy get_sentence_embedding_dimension()
    for method_name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        method = getattr(model, method_name, None)
        if callable(method):
            try:
                val = method()
                # Ensure the return value can be interpreted as a valid dimension
                if isinstance(val, (int, float, np.integer)):
                    return int(val)
            except (TypeError, ValueError):
                continue

    # Fallback to default dimension (384 for all-MiniLM-L6-v2)
    return 384
