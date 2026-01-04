"""Streamlit web interface for the currency classifier.

Designed with accessibility in mind:
- High contrast colors
- Large text for results
- Screen reader friendly labels
- Voice output option
"""

import streamlit as st
import torch
from PIL import Image
from pathlib import Path
import sys
import io

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CurrencyClassifier
from src.utils import load_and_preprocess
from src.voice_output import format_voice_output


# Page config
st.set_page_config(
    page_title="Currency Identifier — Accessibility Tool",
    page_icon="$",
    layout="centered"
)

# Custom CSS for accessibility
st.markdown("""
<style>
    /* High contrast theme */
    .main { background-color: #0a0a1a; }

    .result-box {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ffffff;
        background: #1a1a2e;
        padding: 24px;
        border-radius: 12px;
        border: 2px solid #333366;
        margin: 16px 0;
        line-height: 1.6;
    }

    .confidence-high { color: #00ff88; font-weight: bold; }
    .confidence-medium { color: #ffcc00; font-weight: bold; }
    .confidence-low { color: #ff6b6b; font-weight: bold; }

    .top-predictions {
        background: #16213e;
        padding: 16px;
        border-radius: 8px;
        margin-top: 12px;
    }

    .top-predictions h4 { color: #8888cc; margin-bottom: 8px; }
    .top-predictions p { color: #cccccc; margin: 4px 0; font-size: 1.1rem; }

    /* Make upload area more visible */
    .stFileUploader > div { border: 2px dashed #4444aa !important; }

    /* Large, readable headings */
    h1 { font-size: 2.5rem !important; }
    h2 { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)


# Title and description
st.title("Currency Identifier")
st.write("An accessibility tool that identifies currency denominations from photos of banknotes.")
st.write("Upload a photo or use your camera to identify a banknote.")

st.divider()

# Model loading (cached)
@st.cache_resource
def load_cached_model():
    """Load the trained model (cached across reruns)."""
    model_path = Path(__file__).parent.parent / "models" / "best_model.pth"

    if not model_path.exists():
        return None, None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    # Get class info from checkpoint if available
    num_classes = checkpoint.get("num_classes", 112)
    class_names = checkpoint.get("class_names", None)

    model = CurrencyClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


def run_prediction(image: Image.Image, model, class_names):
    """Run the model on an image and return results."""
    from src.dataset import get_val_transforms

    transform = get_val_transforms()
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    # Get top 3 predictions
    topk_probs, topk_indices = probabilities.topk(3, dim=1)

    if class_names:
        predicted_label = class_names[predicted_idx.item()]
    else:
        predicted_label = f"CLASS_{predicted_idx.item()}"

    # Parse label
    if "_" in predicted_label:
        currency_code, denomination = predicted_label.split("_", 1)
    else:
        currency_code, denomination = predicted_label, "unknown"

    from src.predict import CURRENCY_NAMES
    currency_name = CURRENCY_NAMES.get(currency_code, currency_code)

    result = {
        "currency_code": currency_code,
        "currency_name": currency_name,
        "denomination": denomination,
        "confidence": confidence.item(),
    }

    # Top 3
    top_3 = []
    for prob, idx in zip(topk_probs[0], topk_indices[0]):
        if class_names:
            label = class_names[idx.item()]
        else:
            label = f"CLASS_{idx.item()}"
        top_3.append({"label": label, "confidence": prob.item()})

    result["top_3"] = top_3
    return result


# File upload
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload a banknote image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Take a clear photo of the front of any banknote"
    )

with col2:
    camera_input = st.camera_input(
        "Or take a photo",
        help="Point your camera at a banknote"
    )

# Use whichever input is available
image_source = uploaded_file or camera_input

if image_source:
    # Display the image
    image = Image.open(image_source).convert("RGB")
    st.image(image, caption="Uploaded banknote", use_container_width=True)

    # Load model
    model, class_names = load_cached_model()

    if model is None:
        st.warning("No trained model found. Please train the model first by running: python -m src.train --data-dir <path>")
    else:
        # Run prediction
        with st.spinner("Analyzing banknote..."):
            result = run_prediction(image, model, class_names)

        # Determine confidence level for styling
        conf = result["confidence"]
        if conf >= 0.8:
            conf_class = "confidence-high"
        elif conf >= 0.5:
            conf_class = "confidence-medium"
        else:
            conf_class = "confidence-low"

        # Display result with accessibility markup
        st.markdown(f"""
        <div class="result-box" role="alert" aria-live="polite">
            <strong>Currency:</strong> {result['currency_name']} ({result['currency_code']})<br>
            <strong>Denomination:</strong> {result['denomination']}<br>
            <strong>Confidence:</strong> <span class="{conf_class}">{conf*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Voice output text
        voice_text = format_voice_output(result)
        st.info(f"Voice output: {voice_text}")

        # Top 3 predictions
        if result.get("top_3"):
            st.markdown("""
            <div class="top-predictions">
                <h4>Other possibilities:</h4>
            """, unsafe_allow_html=True)

            for i, pred in enumerate(result["top_3"][1:], 2):  # Skip #1, already shown
                st.markdown(f"""
                <p>{i}. {pred['label']} — {pred['confidence']*100:.1f}%</p>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

else:
    # Instructions when no image is uploaded
    st.markdown("""
    ### How to use:
    1. Take a clear photo of the **front** of any banknote
    2. Upload the image or use your camera
    3. The AI will identify the currency and denomination
    4. Results include a confidence score and voice-ready output

    ### Supported currencies:
    USD, EUR, GBP, INR, JPY, CAD, AUD, CHF, and more.

    ### Tips for best results:
    - Use good lighting
    - Place the banknote on a flat, contrasting surface
    - Capture the entire banknote in the frame
    - Avoid glare and shadows
    """)

# Footer
st.divider()
st.caption("Built as an accessibility tool to help visually impaired people identify currency.")
st.caption("This tool is for educational purposes and should not be used for financial decisions.")
