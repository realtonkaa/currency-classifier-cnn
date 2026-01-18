# Currency Classifier CNN

### An accessibility-focused CNN that identifies currency denominations from photos of banknotes

---

## Why I Built This

My neighbor is blind, and I noticed that whenever she needs to pay for something in cash she has to ask someone around her to tell her what bills she's holding. That kind of dependence on other people for something as basic as money felt wrong to me -- she should be able to just point her phone at a bill and know what it is in seconds.

I also wanted a project where deep learning actually solves a real problem rather than just being a demo. Currency recognition felt like a good fit: the images have clear visual patterns, the classes are well-defined (you know exactly what a $1 looks like vs. a $20), and the output can be read aloud so the app is usable by the people it's meant to help.

---

## What It Does

- Upload a photo of a banknote (or use your phone camera)
- The model identifies the denomination (e.g., $1, $5, $10, $20, $50, $100)
- The result is read aloud using text-to-speech so the user doesn't need to look at the screen
- Simple, accessible web interface built with Streamlit
- Confidence score shown alongside the prediction

---

## Installation

```bash
git clone https://github.com/realtonkaa/currency-classifier-cnn.git
cd currency-classifier-cnn
pip install -r requirements.txt
```

If you want to run the end-to-end tests you'll also need Playwright:

```bash
pip install playwright
playwright install chromium
```

---

## Usage

### Command-line prediction

Classify a single image from the terminal:

```bash
python -m src.predict --image path/to/banknote.jpg
```

Example output:

```
Prediction: USD_20
Confidence: 97.3%
```

You can also pass `--speak` to have the result read aloud:

```bash
python -m src.predict --image path/to/banknote.jpg --speak
```

### Web app

Launch the Streamlit interface:

```bash
streamlit run app/app.py
```

Then open `http://localhost:8501` in your browser. Upload a photo of a banknote and the app will display the prediction and speak it aloud automatically.

### Running tests

```bash
pytest tests/
```

To run only the E2E tests (requires the app to be reachable and Playwright installed):

```bash
pytest tests/test_app_e2e.py -v
```

---

## How It Works

The model is built on **MobileNetV2**, a lightweight CNN architecture originally trained on ImageNet. Rather than training from scratch (which would need millions of images), I use transfer learning: keep the convolutional backbone that already understands edges, textures, and shapes, and replace only the final classification head with a new layer that outputs one score per currency denomination.

Training proceeds in two stages:

1. **Frozen backbone** — only the new classification head is trained for a few epochs. This is fast and gets you to a reasonable baseline quickly without overwriting the pretrained weights.
2. **Full fine-tuning** — the entire network is unfrozen and trained at a lower learning rate. This lets every layer adapt to the specific visual patterns of banknotes and pushes accuracy higher.

MobileNetV2 was chosen because it is small enough to run inference on a mid-range smartphone, which matters for accessibility — the target users may not have the latest hardware.

See `notebooks/02_training_experiments.ipynb` for the full comparison between training from scratch, frozen fine-tuning, and full fine-tuning.

---

## Results

| Model | Validation Accuracy | Training Time |
|---|---|---|
| Basic CNN from scratch | 67% | ~45 min |
| MobileNetV2 (frozen backbone) | 89% | ~20 min |
| MobileNetV2 (full fine-tuning) | **94%** | ~35 min |

The final model reaches **94% validation accuracy** across the supported denominations. Per-class accuracy and confusion matrix are in `notebooks/01_data_exploration.ipynb`.

---

## Limitations

Being honest about what this model doesn't handle well:

- **Heavily worn or damaged bills** — if a note is very crumpled, torn, or defaced, accuracy drops noticeably. The training data skews toward clean, flat bills.
- **Low light and motion blur** — phone photos taken in dim conditions or while moving are harder. Adding blur augmentation during training helped but didn't fully solve it.
- **Partial occlusion** — if a finger covers a significant portion of the note the model sometimes gets confused, especially for denominations that look similar in color (e.g., $1 vs. $5 in USD).
- **Non-US currencies** — the current version focuses on US denominations. Adding other currencies requires more labeled data per currency. It's on the roadmap.
- **Novelty designs** — new series notes (like recent US redesigns) that weren't in the training set may get misclassified until the model is retrained.

---

## What I Learned

A few things that surprised me or that I'd tell myself at the start:

**Transfer learning is almost always the right starting point.** I spent the first week training a basic CNN from scratch, convinced that using pretrained weights would feel like "cheating." That was a mistake. The pretrained MobileNetV2 hit 70% accuracy on the very first epoch. My hand-built CNN took 20 epochs to get there and never surpassed it.

**Data quality matters more than data quantity.** I initially collected a lot of images by scraping, but many were low-resolution thumbnails or had watermarks. Cleaning the dataset and keeping only high-quality photos improved accuracy more than doubling the number of images would have.

**Accessibility is a real constraint, not an afterthought.** Designing for a blind user changed how I thought about the whole interface. I had to think about what happens if the image is bad (give a clear "low confidence" message, not just a wrong answer read aloud), how to handle the TTS across different operating systems, and whether the web layout makes sense when you can't see it.

**Testing a Streamlit app is annoying.** The Playwright E2E tests work, but getting Streamlit's DOM structure to cooperate with selectors took a while. The `data-testid` attributes Streamlit exposes are useful but not always documented.

---

## Tech Stack

- Python 3.9+
- PyTorch + torchvision (model training)
- MobileNetV2 (transfer learning backbone)
- Streamlit (web interface)
- pyttsx3 (text-to-speech / voice output)
- Pillow, NumPy (image processing)
- pytest + Playwright (testing)

---

## License

MIT
