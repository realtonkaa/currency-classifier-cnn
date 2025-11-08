from setuptools import setup, find_packages

setup(
    name="currency-classifier-cnn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "Pillow",
        "numpy",
        "pandas",
        "matplotlib",
        "streamlit",
        "pyttsx3",
        "scikit-learn",
    ],
    python_requires=">=3.9",
)
