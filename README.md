# Text-Similarity-Finder-Using-Embeddings.
Text Similarity Finder using Embeddings

A beginner-friendly Natural Language Processing (NLP) project that compares two text inputs and determines how semantically similar their meanings are using sentence embeddings and cosine similarity.
📖 Table of Contents
1. Project Overview
2. Core NLP Concepts
   - What are Embeddings?
   - Word, Sentence, and Text Embeddings
   - Vectors and Vector Space
   - Semantic Similarity vs. Lexical Similarity
   - Cosine Similarity & Mathematical Formula
3. How the Application Works Step by Step
4. Project Structure
5. Installation & Local Execution (Windows)
6. Running the Unit & Integration Tests
7. Example Test Cases & Interpretation Guide
8. Limitations of Similarity Scores
🚀 Project Overview

Traditional search systems often rely on exact word matching (e.g., does word $X$ appear in document $Y$?). However, human language is nuanced: different words can express the same thought (synonyms), while the exact same words can have different meanings based on context (polysemy).

Text Similarity Finder solves this by converting sentences into numerical representations called dense vector embeddings using a pre-trained Transformer model (`all-MiniLM-L6-v2`). It then computes the cosine similarity between the two vectors to determine their semantic similarity as a percentage, complete with an intuitive Gradio web UI.
🧠 Core NLP Concepts
What are Embeddings?
An embedding is a technique that translates human language (words, sentences, or entire documents) into a list of numbers (a vector) that a machine learning model can process.

Rather than representing words as arbitrary IDs or sparse counts (like Bag-of-Words or One-Hot Encoding), embeddings are dense: every number in the vector encodes a small aspect of linguistic and contextual meaning learned during pre-training on large text corpora.
Word, Sentence, and Text Embeddings
1. Word Embeddings (e.g., Word2Vec, GloVe):
   - Each individual word has its own fixed vector.
   - Limitation: The word "bank" in "river bank" gets the exact same vector as "bank" in "investment bank".
2. Sentence Embeddings (e.g., Sentence Transformers / SBERT):
   - A single vector represents the holistic meaning of an entire sentence.
   - The Transformer model examines the context of surrounding words through self-attention layers and pools them into a single vector.
3. Text / Document Embeddings:
   - Multi-paragraph or full document representations that compress long-form content for document search, clustering, or retrieval-augmented generation (RAG).
     Vectors and Vector Space
     Think of 2D or 3D space:
   - A 2D point is defined by $(x, y)$.
   - A 3D point is defined by $(x, y, z)$.
In machine learning, we represent language in a high-dimensional vector space. The pre-trained model used in this project (`all-MiniLM-L6-v2`) represents every input sentence as a point in a 384-dimensional vector space:

\(\mathbf{v} = [v_1, v_2, v_3, \dots, v\_{384}]\)

In this vector space:
- Sentences with similar meanings are positioned close to one another and point in almost the same direction.
- Sentences about unrelated topics point in different directions.
```
Vector Space Intuition (Projected to 2D)

Text B: "A fast puppy is playing outside."
^
/  \  Small angle (High Similarity)
/    
Text A: "A dog is running in the yard."


\  Large angle (Low Similarity)
v
Text C: "Quantum computing uses qubits."
```
Semantic Similarity vs. Lexical Similarity
- Lexical Similarity: Measures whether two sentences share the same literal words and spelling (e.g., Jaccard similarity, Levenshtein distance).
- Semantic Similarity: Measures whether two sentences convey the same conceptual meaning, even if they do not share any words in common!
Example:
- Sentence 1: "The physician examined the sick child."
- Sentence 2: "The doctor checked the ill kid."
- Shared words: 0 (except "the")
- Lexical similarity: Very low (~0%)
- Semantic similarity: Extremely high (~92%)
  Cosine Similarity & Mathematical Formula
  Cosine similarity measures the cosine of the angle ($\theta$) between two non-zero vectors $\mathbf{A}$ and $\mathbf{B}$ in high-dimensional space:
\(\text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\\|\mathbf{A}\\|\_2 \\|\mathbf{B}\\|\_2} = \frac{\sum\_{i=1}^{n} A_i B_i}{\sqrt{\sum\_{i=1}^{n} A_i^2} \cdot \sqrt{\sum\_{i=1}^{n} B_i^2}}\)
Why Cosine Similarity instead of Euclidean Distance?
- Magnitude Invariance: Euclidean distance ($L_2$ distance) measures the straight-line distance between vector tips. If one text is very short and another is longer, their vector magnitudes might differ, yielding an artificially large Euclidean distance.
- Directional Focus: Cosine similarity only measures vector orientation/direction, not length. Two vectors pointing in the exact same direction have $\cos(0^\circ) = 1.0$, regardless of their individual lengths.
  Mathematical Range:
- $\cos(\theta) = 1.0$ (100%): Identical direction (identical semantic meaning).
- $\cos(\theta) = 0.0$ (0%): Orthogonal vectors (completely independent/unrelated concepts).
- $\cos(\theta) = -1.0$: Directly opposite vectors (rare in semantic language models, mapped to 0%).
🔄 How the Application Works Step by Step

```
+-------------------------------------------------------------------+
|                           User Input                              |
|           Text 1: "The cat slept on the mat."                     |
|           Text 2: "A feline was napping on the rug."              |
+---------------------------------+---------------------------------+
|
v
+-------------------------------------------------------------------+
|                     1. Input Validation                           |
|        Check non-empty strings and strip whitespace               |
+---------------------------------+---------------------------------+
|
v
+-------------------------------------------------------------------+
|               2. Tokenization & Model Encoding                    |
|             (Pre-trained: all-MiniLM-L6-v2)                       |
|   Pass tokens through Transformer layers and apply Mean Pooling   |
+---------------------------------+---------------------------------+
|
v
+-------------------------------------------------------------------+
|                 3. Dense Numerical Embeddings                     |
|         Vector A: [0.042, -0.015, ..., 0.098]  (384 numbers)      |
|         Vector B: [0.039, -0.018, ..., 0.102]  (384 numbers)      |
+---------------------------------+---------------------------------+
|
v
+-------------------------------------------------------------------+
|           4. Cosine Similarity Calculation (Scikit-Learn)         |
|              Compute dot product divided by norm product          |
|              Score = 0.8841  -->  Percentage = 88.41%             |
+---------------------------------+---------------------------------+
|
v
+-------------------------------------------------------------------+
|             5. Nuanced Interpretation & Visual UI                 |
|   Category: "Very High Similarity" | Color: Green                 |
|   Visual Progress Bar Meter + Detailed Metric Breakdown           |
+-------------------------------------------------------------------+
```
1. Model Loading (Singleton): The application loads `all-MiniLM-L6-v2` once during startup and caches it in memory. Subsequent comparisons run immediately without reloading the model.
2. Input Validation: Both inputs are checked to ensure they are non-empty and not just spaces.
3. Embedding Generation: Each text is passed into `model.encode()`, returning a 384-dimensional vector.
4. Cosine Computation: Scikit-learn's `cosine_similarity` calculates the angle between both vectors.
5. Percentage Formatting & Interpretation: The raw float score is transformed into a percentage and assigned a nuanced interpretation level.
6. UI Rendering: Gradio displays the similarity percentage, color-coded level badge, a visual progress meter, and a technical breakdown.
📁 Project Structure

```
Text Similarity Finder using Embeddings/
│
├── app.py                  # Gradio user interface and comparison workflow
├── embeddings.py           # Model loading, caching, and embedding generation
├── similarity.py           # Cosine similarity math, score conversion, and interpretation
├── requirements.txt        # Project dependencies (Sentence-Transformers, Scikit-learn, Gradio, etc.)
├── README.md               # Complete project documentation and guide
│
└── tests/
├── __init__.py         # Test package marker
└── test_similarity.py # Unit tests (math, validation, mocks) & integration tests
```
💻 Installation & Local Execution (Windows)

Follow these step-by-step instructions in Windows PowerShell or Command Prompt:
1. Open Terminal and Navigate to Project Directory
   ```powershell
   cd "c:\Users\Metapi\OneDrive\Documents\Text Similarity Finder using Embeddings"
   ```
2. (Optional but Recommended) Create and Activate a Virtual Environment
   ```powershell
   Create a virtual environment named 'venv'
   python -m venv venv
   Activate the virtual environment
   In PowerShell:
   .\venv\Scripts\Activate.ps1
   (If using CMD instead of PowerShell):
   .\venv\Scripts\activate.bat
   ```
Note for PowerShell users: If you receive an execution policy script error when activating, run:
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`
3. Install Required Dependencies
   ```powershell
   pip install -r requirements.txt
   ```
4. Launch the Gradio Web Application
   ```powershell
   python app.py
   ```
5. Open the Interface
   Open your web browser and navigate to:
   ```
   http://127.0.0.1:7860
   ```

🧪 Running the Unit & Integration Tests

The project includes a comprehensive test suite in `tests/test_similarity.py` covering:
- Exact mathematical properties of cosine similarity (identical, orthogonal, opposite, scaled vectors, known 45° angles).
- Zero vectors, dimension mismatches, and empty inputs.
- Percentage rounding and boundary handling.
- Input validation checks.
- Mocked embedding generation to guarantee fast, offline testing without downloading models.
- An integration test ensuring semantic pairs score significantly higher than unrelated pairs.
  Run All Tests:
  ```powershell
  pytest -v
  ```
  Run Only Fast Math & Unit Tests:
  ```powershell
  pytest -v -k "not Integration"
  ```
📊 Example Test Cases & Interpretation Guide
Example Pair	Text 1	Text 2	Expected Similarity	Interpretation Category
Identical	"Machine learning powers modern AI."	"Machine learning powers modern AI."	100.00%	Very High Similarity
Paraphrased	"The quick brown fox jumps over the lazy dog."	"A fast auburn fox leaps over a sleepy canine."	~88% – 93%	Very High Similarity
Topical Overlap	"AI is transforming healthcare with disease detection."	"Doctors use machine learning algorithms to diagnose scans."	~75% – 85%	High Similarity
Same Domain, Opposite Sentiment	"I loved the food at this restaurant; it was delicious."	"The food was awful and completely tasteless."	~55% – 65%	Moderate Similarity (shares food/dining context)
Completely Unrelated	"The planets orbit around the Sun."	"Baking sourdough requires flour, salt, and water."	~5% – 20%	Very Low / Unrelated
Score Interpretation Reference:				


- $\ge 88\%$ — Very High Similarity: Both inputs share essentially identical meaning or close paraphrasing.
- $70\% - 87.9\%$ — High Similarity: Strong thematic alignment and significant conceptual overlap.
- $45\% - 69.9\%$ — Moderate Similarity: The texts share common domain topics or vocabulary, but express different specific ideas.
- $20\% - 44.9\%$ — Low Similarity: Weak or incidental topical connection.
- $< 20\%$ — Very Low / Unrelated: Distinct conceptual domains with virtually no semantic relation.
⚠️ Limitations of Similarity Scores

While Sentence Transformers are powerful, beginners should understand their natural limitations:
1. Negation Sensitivity:
   - Pre-trained embeddings excel at topical categorization, but sometimes struggle with subtle logical negation.
   - For example, "I like pizza" and "I do not like pizza" might score relatively high (~60–70%) because both texts are about pizza preference. The model recognizes strong topical affinity even if the logical truth value is inverted.
2. Factual Correctness vs. Meaning:
   - Embeddings measure semantic similarity, not factual accuracy.
   - "The Earth orbits the Sun" and "The Sun orbits the Earth" will score very high because the vocabulary and conceptual domain are identical, even though one is astronomically false.
3. No External Knowledge Base:
   - Similarity calculation evaluates sentence vector orientation only; it does not consult a database, search the web, or perform formal reasoning.
4. Length Disparities:
   - Comparing a single word against a multi-paragraph article can dilute the vector representation during mean-pooling, potentially reducing the apparent similarity score.
5. Score Calibration:
   - A cosine similarity of `0.60` does not mean the texts are "60% true". It is a relative geometric measure, which is why qualitative interpretation bands are essential.
