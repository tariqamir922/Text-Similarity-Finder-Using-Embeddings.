"""
app.py
======
Interactive Gradio web interface for the Text Similarity Finder.

This application compares two text inputs and calculates their semantic similarity
using the 'all-MiniLM-L6-v2' Sentence Transformer model and Scikit-learn's
cosine similarity metric.
"""

import sys
from typing import Tuple

# Ensure Windows terminal standard streams handle UTF-8 characters without errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import gradio as gr

from embeddings import load_embedding_model, generate_embedding, get_embedding_dimension
from similarity import (
    compute_cosine_similarity,
    score_to_percentage,
    interpret_similarity,
    validate_input_pair,
)

# -----------------------------------------------------------------------------
# 1. Model Initialization
# -----------------------------------------------------------------------------
# Load the model once at application startup to ensure fast responses during inference.
print("Initializing sentence transformer model ('all-MiniLM-L6-v2')...")
try:
    embedding_model = load_embedding_model()
    model_dim = get_embedding_dimension(embedding_model)
    print(f"Model successfully loaded. Embedding dimension: {model_dim}")
except Exception as error:
    print(f"CRITICAL ERROR: Failed to load model at startup:\n{error}", file=sys.stderr)
    embedding_model = None
    model_dim = 384


# -----------------------------------------------------------------------------
# 2. Core Comparison Workflow
# -----------------------------------------------------------------------------
def run_similarity_analysis(
    text1: str, text2: str
) -> Tuple[str, str, str, str]:
    """
    Coordinates the text similarity workflow:
    1. Validates user inputs.
    2. Encodes both texts into 384-dimensional dense vectors.
    3. Calculates the cosine similarity between the vectors.
    4. Formats the score into a percentage and generates an interpretation.
    5. Returns formatted HTML and markdown components for Gradio display.

    Returns:
        Tuple containing:
        - alert_html (str): Validation message or status banner.
        - score_html (str): Highlighted score card with color-coded badge.
        - progress_html (str): Visual progress bar meter.
        - details_markdown (str): Detailed interpretation and technical breakdown.
    """
    # Verify the model is operational
    if embedding_model is None:
        error_msg = (
            "<div style='background-color: #fee2e2; border-left: 5px solid #dc2626; "
            "padding: 14px; border-radius: 8px; color: #991b1b; margin-bottom: 12px;'>"
            "<strong>[!] Model Error:</strong> The sentence embedding model could not be initialized. "
            "Please check your internet connection or install missing dependencies."
            "</div>"
        )
        return error_msg, "", "", ""

    # Validate inputs
    is_valid, validation_msg = validate_input_pair(text1, text2)
    if not is_valid:
        gr.Warning(validation_msg)
        warning_html = (
            f"<div style='background-color: #fffbeb; border-left: 5px solid #f59e0b; "
            f"padding: 14px; border-radius: 8px; color: #92400e; margin-bottom: 12px;'>"
            f"<strong>[!] Validation Notice:</strong> {validation_msg}"
            f"</div>"
        )
        empty_meter = (
            "<div style='background-color: #f3f4f6; border-radius: 9999px; height: 18px; "
            "width: 100%; overflow: hidden; margin: 10px 0;'>"
            "<div style='background-color: #d1d5db; width: 0%; height: 100%;'></div>"
            "</div>"
        )
        return warning_html, "<p style='color:#6b7280;'>Awaiting valid input texts...</p>", empty_meter, ""

    try:
        # Step A: Convert text into 384-dimensional dense numerical vectors
        emb1 = generate_embedding(text1, model=embedding_model)
        emb2 = generate_embedding(text2, model=embedding_model)

        # Step B: Calculate cosine similarity between vectors
        raw_cosine = compute_cosine_similarity(emb1, emb2)

        # Step C: Convert cosine score [-1.0, 1.0] to a percentage [0.0%, 100.0%]
        percentage = score_to_percentage(raw_cosine)

        # Step D: Retrieve nuanced educational interpretation
        interp = interpret_similarity(percentage)

    except Exception as exc:
        gr.Error(f"Comparison error: {str(exc)}")
        err_box = (
            f"<div style='background-color: #fee2e2; border-left: 5px solid #dc2626; "
            f"padding: 14px; border-radius: 8px; color: #991b1b; margin-bottom: 12px;'>"
            f"<strong>Calculation Error:</strong> {str(exc)}"
            f"</div>"
        )
        return err_box, "", "", ""

    # Build the clean, modern visual components
    status_html = (
        "<div style='background-color: #ecfdf5; border-left: 5px solid #10b981; "
        "padding: 10px 14px; border-radius: 6px; color: #065f46; font-size: 0.95em; margin-bottom: 12px;'>"
        "<strong>[OK] Success:</strong> Embeddings generated and semantic similarity calculated."
        "</div>"
    )

    score_card_html = f"""
    <div style='background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px;
                text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.04); margin-bottom: 14px;'>
        <div style='font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600;'>
            Semantic Similarity Score
        </div>
        <div style='font-size: 3rem; font-weight: 800; color: {interp["color"]}; margin: 6px 0; line-height: 1.1;'>
            {percentage:.2f}%
        </div>
        <div style='display: inline-block; background-color: {interp["color"]}1a; color: {interp["color"]};
                    font-weight: 700; padding: 4px 14px; border-radius: 9999px; font-size: 0.95em;'>
            {interp["badge"]}
        </div>
    </div>
    """

    progress_meter_html = f"""
    <div style='margin-bottom: 16px;'>
        <div style='display: flex; justify-content: space-between; font-size: 0.82em; color: #64748b; margin-bottom: 4px; font-weight: 600;'>
            <span>0% (Completely Unrelated)</span>
            <span>50% (Partial Overlap)</span>
            <span>100% (Identical Meaning)</span>
        </div>
        <div style='background-color: #e2e8f0; border-radius: 9999px; height: 18px; width: 100%;
                    overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);'>
            <div style='background: linear-gradient(90deg, {interp["color"]}cc, {interp["color"]});
                        width: {max(2.0, percentage)}%; height: 100%; border-radius: 9999px;
                        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);'></div>
        </div>
    </div>
    """

    # Prepare word and character metrics
    t1_words = len(text1.strip().split())
    t2_words = len(text2.strip().split())

    details_md = f"""
### Semantic Interpretation
> **{interp['summary']}**

{interp['details']}

---

### Technical Analysis
| Metric | Value |
| :--- | :--- |
| **Cosine Similarity (cos θ)** | `{raw_cosine:.4f}` *(Range: -1.0 to +1.0)* |
| **Similarity Percentage** | `{percentage:.2f}%` |
| **Embedding Dimensions** | `384 features per text` |
| **Model Name** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Text 1 Length** | `{t1_words} words ({len(text1.strip())} chars)` |
| **Text 2 Length** | `{t2_words} words ({len(text2.strip())} chars)` |

> **Beginner Note on Semantic Similarity:**
> Sentence embeddings map overall conceptual topics into high-dimensional space.
> A high score indicates that the concepts and vocabulary contexts are closely related,
> even when opposite sentiments are expressed (e.g., *"I love tea"* vs *"I hate tea"* both talk about tea preference).
"""

    return status_html, score_card_html, progress_meter_html, details_md


def clear_inputs() -> Tuple[str, str, str, str, str, str]:
    """Clears all text input fields and resets the output panels."""
    empty_meter = (
        "<div style='background-color: #f3f4f6; border-radius: 9999px; height: 18px; "
        "width: 100%; overflow: hidden; margin: 10px 0;'>"
        "<div style='background-color: #d1d5db; width: 0%; height: 100%;'></div>"
        "</div>"
    )
    return (
        "",  # text1
        "",  # text2
        "",  # alert_status
        "<p style='color:#6b7280; text-align: center; padding: 20px;'>Enter two texts and click <strong>Compare Texts</strong> to view results.</p>",
        empty_meter,
        "",  # details
    )


# -----------------------------------------------------------------------------
# 3. Gradio Interface Construction
# -----------------------------------------------------------------------------
custom_css = """
footer {visibility: hidden !important;}
.gradio-container {max-width: 1100px !important; margin: 0 auto !important;}
"""

with gr.Blocks(title="Text Similarity Finder using Embeddings") as demo:
    gr.Markdown(
        """
        # Text Similarity Finder using Embeddings
        Compare the semantic meaning of two sentences or paragraphs using **Sentence Transformers** (`all-MiniLM-L6-v2`) and **Cosine Similarity**.
        Unlike simple keyword matching, sentence embeddings capture the underlying *meaning* of the texts.
        """
    )

    with gr.Row():
        # Left Column: User Inputs & Controls
        with gr.Column(scale=5):
            gr.Markdown("### Input Texts")
            input_text1 = gr.Textbox(
                label="Text 1",
                placeholder="Enter the first sentence or paragraph here...",
                lines=4,
                max_lines=8,
            )
            input_text2 = gr.Textbox(
                label="Text 2",
                placeholder="Enter the second sentence or paragraph here to compare...",
                lines=4,
                max_lines=8,
            )

            with gr.Row():
                btn_compare = gr.Button("Compare Texts", variant="primary", size="lg")
                btn_clear = gr.Button("Clear", variant="secondary", size="lg")

            # Pre-configured beginner test examples
            gr.Markdown("#### Try an Example:")
            examples = [
                [
                    "The quick brown fox jumps over the lazy dog.",
                    "A fast auburn fox leaps over a sleepy canine.",
                ],
                [
                    "Artificial intelligence is transforming healthcare with early disease detection.",
                    "Doctors are using machine learning models to analyze medical scans with high accuracy.",
                ],
                [
                    "I really enjoyed the pasta at this restaurant; the flavors were incredible.",
                    "The food was terrible and the pasta was cold and completely flavorless.",
                ],
                [
                    "The solar system consists of eight planets orbiting the Sun.",
                    "Baking a traditional sourdough loaf requires sourdough starter, flour, and water.",
                ],
            ]
            gr.Examples(
                examples=examples,
                inputs=[input_text1, input_text2],
                label="Click an example to load it into the text boxes:",
            )

        # Right Column: Visual Similarity Results
        with gr.Column(scale=5):
            gr.Markdown("### Similarity Results")
            out_alert = gr.HTML(value="")
            out_score_card = gr.HTML(
                value="<p style='color:#6b7280; text-align: center; padding: 20px;'>Enter two texts and click <strong>Compare Texts</strong> to view results.</p>"
            )
            out_progress = gr.HTML(
                value="<div style='background-color: #f3f4f6; border-radius: 9999px; height: 18px; width: 100%; overflow: hidden; margin: 10px 0;'><div style='background-color: #d1d5db; width: 0%; height: 100%;'></div></div>"
            )
            out_details = gr.Markdown(value="")

    # Educational Accordion
    with gr.Accordion("Beginner's Guide: How Semantic Similarity Works", open=False):
        gr.Markdown(
            """
            ### What happens when you click "Compare Texts"?
            1. **Tokenization:** Each text input is split into subwords and numerical token IDs.
            2. **Deep Neural Encoding:** The `all-MiniLM-L6-v2` Transformer processes tokens and evaluates context across attention layers.
            3. **Mean Pooling:** The representations across all tokens are averaged into a single **384-dimensional dense vector**.
            4. **Cosine Similarity Calculation:**
               Using Scikit-learn, the cosine of the angle $\\theta$ between vector $\\mathbf{A}$ and vector $\\mathbf{B}$ is computed:
               $$\\text{Cosine Similarity}(\\mathbf{A}, \\mathbf{B}) = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|}$$
            5. **Score & Percentage:** The resulting cosine score (typically 0.0 to 1.0) is converted to a percentage and evaluated against semantic thresholds.
            """
        )

    # Wire up button event listeners
    btn_compare.click(
        fn=run_similarity_analysis,
        inputs=[input_text1, input_text2],
        outputs=[out_alert, out_score_card, out_progress, out_details],
    )

    btn_clear.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[input_text1, input_text2, out_alert, out_score_card, out_progress, out_details],
    )


# -----------------------------------------------------------------------------
# 4. Application Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Launching Gradio application...")
    # launch() runs the local development web server.
    # Access the interface at http://127.0.0.1:7860
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=custom_css,
    )
