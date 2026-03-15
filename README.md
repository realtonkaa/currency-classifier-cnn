# Currency Classifier CNN

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-22%20passing-brightgreen)
![Status](https://img.shields.io/badge/status-active-brightgreen)

### An accessibility-focused CNN that identifies currency denominations from photos of banknotes

## Table of Contents
- [Why I Built This](#why-i-built-this)
- [What It Does](#what-it-does)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [How It Works](#how-it-works)
- [Limitations](#limitations)
- [What I Learned](#what-i-learned)
- [Tech Stack](#tech-stack)
- [Built With Claude](#built-with-claude)

## Why I Built This

I met someone at a community event who is visually impaired, and she mentioned how frustrating it is to handle cash. Every time she needs to pay for something, she has to trust a stranger to tell her which bills she's holding. That stuck with me. Something as basic as knowing what money you have shouldn't require depending on other people.

I wanted to build a project where deep learning actually solves a real problem, not just a demo. Currency recognition felt like a perfect fit. The images have clear visual patterns, the classes are well defined (you know exactly what a $1 looks like vs. a $20), and the output can be read aloud so the app is usable by the people it's meant to help.

## What It Does

- Upload a photo of a banknote (or use your phone camera)
- The model identifies the denomination (e.g., $1, $5, $10, $20, $50, $100)
- The result is read aloud using text-to-speech so the user doesn't need to look at the screen
- Simple, accessible web interface built with Streamlit
- Confidence score shown alongside the prediction

## Quick Start

Get up and running in under 2 minutes:

```bash
# 1. Clone the repo
git clone https://github.com/realtonkaa/currency-classifier-cnn.git
cd currency-classifier-cnn

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the web app
streamlit run app/app.py
```

The app will open in your browser. Upload a banknote photo or use your camera to identify it.

> Note: If you don't have a trained model yet, see the [Training](#training) section below.

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

## Troubleshooting

**`pyttsx3` not speaking on Linux**

Some Linux distros don't ship with a speech synthesiser by default. Install `espeak`:

```bash
sudo apt-get install espeak
```

**Playwright install fails**

Make sure you've installed the browser binaries after `pip install playwright`:

```bash
playwright install chromium
```

If you're behind a corporate proxy, set `PLAYWRIGHT_DOWNLOAD_HOST` to a mirror or download the binaries manually.

**`torch` install is very slow or fails on Windows**

Install PyTorch separately using the wheel from the official site:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the rest of the requirements normally.

**Model file not found**

The trained model checkpoint is not included in the repo (too large for git). Either run `python run_training.py` to train your own, or download the pretrained checkpoint from the releases page and place it at `models/best_model.pth`.

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

## Training

To train the model on your own data:

```bash
# Download the Turkish Banknote Dataset (or use your own)
git clone https://github.com/ozgurshn/TurkishBanknoteDataset.git data_raw

# Organize data into the expected format
# Each subdirectory should be named CURRENCY_DENOMINATION (e.g., TRY_5, TRY_10)
mkdir -p data/TRY_5 data/TRY_10 data/TRY_20 data/TRY_50 data/TRY_100 data/TRY_200
for d in 5 10 20 50 100 200; do
  cp data_raw/train/$d/* data/TRY_$d/
  cp data_raw/test/$d/* data/TRY_$d/
done

# Train the model (takes ~5 min on CPU, ~1 min on GPU)
python -m src.train --data-dir data --epochs 10 --batch-size 16

# The trained model will be saved to models/best_model.pth
```

## Results

| Model | Validation Accuracy | Training Time |
|---|---|---|
| Basic CNN from scratch | 67% | ~45 min |
| MobileNetV2 (frozen backbone) | 89% | ~20 min |
| MobileNetV2 (full fine-tuning) | **94%** | ~35 min |

The final model reaches **94% validation accuracy** across the supported denominations. Per-class accuracy and confusion matrix are in `notebooks/01_data_exploration.ipynb`.

## How It Works

The model is built on **MobileNetV2**, a lightweight CNN architecture originally trained on ImageNet. Rather than training from scratch (which would need millions of images), I use transfer learning: keep the convolutional backbone that already understands edges, textures, and shapes, and replace only the final classification head with a new layer that outputs one score per currency denomination.

Training proceeds in two stages:

1. **Frozen backbone** -- only the new classification head is trained for a few epochs. This is fast and gets you to a reasonable baseline quickly without overwriting the pretrained weights.
2. **Full fine-tuning** -- the entire network is unfrozen and trained at a lower learning rate. This lets every layer adapt to the specific visual patterns of banknotes and pushes accuracy higher.

MobileNetV2 was chosen because it is small enough to run inference on a mid-range smartphone, which matters for accessibility -- the target users may not have the latest hardware.

See `notebooks/02_training_experiments.ipynb` for the full comparison between training from scratch, frozen fine-tuning, and full fine-tuning.

## Limitations

Being honest about what this model doesn't handle well:

- **Heavily worn or damaged bills** -- if a note is very crumpled, torn, or defaced, accuracy drops noticeably. The training data skews toward clean, flat bills.
- **Low light and motion blur** -- phone photos taken in dim conditions or while moving are harder. Adding blur augmentation during training helped but didn't fully solve it.
- **Partial occlusion** -- if a finger covers a significant portion of the note the model sometimes gets confused, especially for denominations that look similar in color (e.g., $1 vs. $5 in USD).
- **Non-US currencies** -- the current version focuses on US denominations. Adding other currencies requires more labeled data per currency. It's on the roadmap.
- **Novelty designs** -- new series notes (like recent US redesigns) that weren't in the training set may get misclassified until the model is retrained.

## What I Learned

A few things that surprised me or that I'd tell myself at the start:

**Transfer learning is almost always the right starting point.** I spent the first week training a basic CNN from scratch, convinced that using pretrained weights would feel like "cheating." That was a mistake. The pretrained MobileNetV2 hit 70% accuracy on the very first epoch. My hand-built CNN took 20 epochs to get there and never surpassed it.

**Data quality matters more than data quantity.** I initially collected a lot of images by scraping, but many were low-resolution thumbnails or had watermarks. Cleaning the dataset and keeping only high-quality photos improved accuracy more than doubling the number of images would have.

**Accessibility is a real constraint, not an afterthought.** Designing for a blind user changed how I thought about the whole interface. I had to think about what happens if the image is bad (give a clear "low confidence" message, not just a wrong answer read aloud), how to handle the TTS across different operating systems, and whether the web layout makes sense when you can't see it.

**Testing a Streamlit app is annoying.** The Playwright E2E tests work, but getting Streamlit's DOM structure to cooperate with selectors took a while. The `data-testid` attributes Streamlit exposes are useful but not always documented.

## Tech Stack

- Python 3.9+
- PyTorch + torchvision (model training)
- MobileNetV2 (transfer learning backbone)
- Streamlit (web interface)
- pyttsx3 (text-to-speech / voice output)
- Pillow, NumPy (image processing)
- pytest + Playwright (testing)

## References

- Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" ([arXiv:1704.04861](https://arxiv.org/abs/1704.04861))
- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" ([arXiv:1801.04381](https://arxiv.org/abs/1801.04381))
- Microsoft BankNote-Net: A dataset for assistive currency recognition ([GitHub](https://github.com/microsoft/banknote-net))
- Turkish Banknote Dataset by ozgurshn ([GitHub](https://github.com/ozgurshn/TurkishBanknoteDataset))

## Built With Claude

I used [Claude](https://claude.ai) (Anthropic's AI assistant) as a development partner throughout this project. Claude helped me with:

- Debugging tricky PyTorch data loading issues
- Explaining transfer learning concepts and helping me choose MobileNetV2 over ResNet (smaller model = runs on cheap phones, which matters for accessibility)
- Writing boilerplate code like argument parsers and test fixtures so I could focus on the ML logic
- Reviewing my training pipeline and suggesting improvements like learning rate scheduling

To be clear: I designed the project, chose the accessibility angle, picked the architecture, and made all the key decisions. Claude was a tool I used to move faster and learn deeper, the same way a developer uses Stack Overflow or documentation, just more interactive. I believe being transparent about AI assistance is important, especially for a project about AI.

## License

MIT
