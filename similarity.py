"""
similarity.py
=============
Calculates cosine similarity between embedding vectors, formats the similarity
score as a percentage, and provides contextual, beginner-friendly interpretations.

Mathematical Concept:
---------------------
Cosine similarity measures the cosine of the angle between two multi-dimensional
vectors in vector space:

                A · B              Σ (A_i * B_i)
    Cosine = ----------- = -------------------------------
             ||A|| ||B||    sqrt(Σ A_i^2) * sqrt(Σ B_i^2)

- If two texts have identical meanings, their vectors point in the same direction:
  Angle = 0°, cos(0°) = 1.0 (100% similarity).
- If two texts are completely orthogonal (unrelated concepts in different domains):
  Angle = 90°, cos(90°) = 0.0 (0% similarity).
- If vectors point in opposite directions:
  Angle = 180°, cos(180°) = -1.0 (-100% similarity, rare in semantic models).
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two 1D or 2D numerical vectors using Scikit-Learn.

    Parameters:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score bounded strictly within [-1.0, 1.0].

    Raises:
        ValueError: If input vectors are empty, have different lengths,
                    or contain only zeros (norm is zero).
    """
    # Convert inputs to standard numpy float arrays
    v1 = np.asarray(vec1, dtype=np.float32).flatten()
    v2 = np.asarray(vec2, dtype=np.float32).flatten()

    if v1.size == 0 or v2.size == 0:
        raise ValueError("Cannot calculate cosine similarity for empty vectors.")

    if v1.shape != v2.shape:
        raise ValueError(
            f"Vector dimensions do not match: {v1.shape[0]} vs {v2.shape[0]}."
        )

    # Check for zero vectors (all elements are 0), which would cause division by zero
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Cannot calculate similarity for a zero vector (norm is zero).")

    # Reshape vectors to 2D arrays (1, N) as required by Scikit-learn's cosine_similarity
    v1_2d = v1.reshape(1, -1)
    v2_2d = v2.reshape(1, -1)

    # Calculate similarity using scikit-learn
    raw_similarity = float(cosine_similarity(v1_2d, v2_2d)[0][0])

    # Clamp the result to [-1.0, 1.0] to eliminate floating-point rounding quirks (e.g. 1.0000000000000002)
    clamped_similarity = max(-1.0, min(1.0, raw_similarity))
    return clamped_similarity


def score_to_percentage(similarity_score: float) -> float:
    """
    Converts a cosine similarity score (typically between 0.0 and 1.0 for sentence embeddings)
    into a human-readable percentage rounded to two decimal places.

    Negative cosine values (if any) are bounded at 0.0% since they represent
    diametrically opposed or non-similar content in semantic vector space.

    Parameters:
        similarity_score (float): The raw cosine score in range [-1.0, 1.0].

    Returns:
        float: Similarity percentage between 0.0% and 100.0%.
    """
    bounded_score = max(0.0, min(1.0, float(similarity_score)))
    return round(bounded_score * 100.0, 2)


def interpret_similarity(percentage: float) -> Dict[str, Any]:
    """
    Provides a nuanced, educational interpretation of a semantic similarity percentage.

    Important Note on Similarity Interpretation:
    --------------------------------------------
    A high score does NOT necessarily mean the texts are identical or agree factually!
    For example:
    - "I love apple pie" and "I hate apple pie" share high similarity because both
      revolve around eating desserts and apples (same topic), despite opposite sentiment.
    - Semantic similarity reflects shared conceptual domain and vocabulary context,
      rather than logical equivalence or truth value.

    Parameters:
        percentage (float): Similarity percentage between 0.0 and 100.0.

    Returns:
        dict: A dictionary containing:
            - 'level': Short category descriptor (e.g., 'Very High', 'Moderate').
            - 'color': UI display color code.
            - 'summary': Clear one-line summary.
            - 'details': In-depth educational explanation.
    """
    if percentage >= 88.0:
        return {
            "level": "Very High Similarity",
            "color": "#16a34a",  # Emerald Green
            "badge": "Very High Similarity",
            "summary": "These texts convey nearly identical core meanings or close paraphrasing.",
            "details": (
                "The two inputs share virtually the same intent, concepts, and topical structure. "
                "Any differences are likely minor phrasing variations or synonyms."
            ),
        }
    elif percentage >= 70.0:
        return {
            "level": "High Similarity",
            "color": "#2563eb",  # Blue
            "badge": "High Similarity",
            "summary": "These texts share a strong thematic connection and significant conceptual overlap.",
            "details": (
                "Both statements discuss the same subject matter from closely related perspectives, "
                "though they may include different specific details or phrasing."
            ),
        }
    elif percentage >= 45.0:
        return {
            "level": "Moderate Similarity",
            "color": "#d97706",  # Amber
            "badge": "Moderate Similarity",
            "summary": "These texts have partial topical overlap or share broad domain themes.",
            "details": (
                "The sentences may relate to the same overall domain (e.g., technology or food), "
                "but they convey noticeably different messages or focus on different aspects."
            ),
        }
    elif percentage >= 20.0:
        return {
            "level": "Low Similarity",
            "color": "#ea580c",  # Orange
            "badge": "Low Similarity",
            "summary": "These texts have minimal semantic relation.",
            "details": (
                "There may be an incidental shared word or very distant category link, "
                "but the statements address largely different topics."
            ),
        }
    else:
        return {
            "level": "Very Low / Unrelated",
            "color": "#dc2626",  # Red
            "badge": "Very Low / Unrelated",
            "summary": "These texts are conceptually unrelated.",
            "details": (
                "The vectors point in substantially different directions in semantic vector space. "
                "The inputs represent completely distinct domains or concepts."
            ),
        }


def validate_input_pair(text1: Any, text2: Any) -> Tuple[bool, str]:
    """
    Validates whether both inputs are non-empty strings.

    Parameters:
        text1: First input value.
        text2: Second input value.

    Returns:
        tuple (is_valid: bool, error_message: str):
            - If valid: (True, "")
            - If invalid: (False, descriptive_reason)
    """
    if text1 is None or text2 is None:
        return False, "Both text inputs must be provided."

    if not isinstance(text1, str) or not isinstance(text2, str):
        return False, "Both inputs must be text strings."

    t1_clean = text1.strip()
    t2_clean = text2.strip()

    if not t1_clean and not t2_clean:
        return False, "Both Text 1 and Text 2 are empty. Please enter text in both fields."
    if not t1_clean:
        return False, "Text 1 is empty. Please enter text for the first input."
    if not t2_clean:
        return False, "Text 2 is empty. Please enter text for the second input."

    return True, ""
