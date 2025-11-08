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

*Coming soon -- will add once the training pipeline is working.*

---

## Usage

*Coming soon.*

---

## How It Works

*I'll write this up after I finish training. Short version: MobileNetV2 pretrained on ImageNet, fine-tuned on banknote images, with a custom classification head.*

---

## Results

*Not done training yet. Will post accuracy, confusion matrix, and some example predictions here.*

---

## Limitations

*Will fill this in honestly once I know what the model struggles with. Guessing: crumpled bills, low light, partial occlusion.*

---

## What I Learned

*Will write this up at the end of the project.*

---

## Tech Stack

- Python 3.9+
- PyTorch + torchvision (model training)
- MobileNetV2 (transfer learning backbone)
- Streamlit (web interface)
- pyttsx3 (text-to-speech / voice output)
- Pillow, NumPy (image processing)
- pytest + Playwright (testing)
