# AI Handoff Document: SmartScan AI

Welcome to the SmartScan AI repository! This document is designed to quickly onboard any AI assistant (like Claude, ChatGPT, Gemini, or GitHub Copilot) to the current state, architecture, and constraints of this project.

## 1. Project Overview
**SmartScan AI** is a Digital Image Processing (DIP) mini-project. It turns a photo of a physical document into a clean, scanner-like digital image using **classical image-processing techniques** (no deep learning/black-box models for the core pipeline).

The goal is to demonstrate: image enhancement, noise removal, segmentation, thresholding, morphological operations, geometric transforms, and image compression.

## 2. Current State (August 2026)
We have successfully completed Phases 1 through 8.
- **Core Pipeline (`src/` & `main.py`)**: Fully functional. It detects documents, applies perspective warps, removes shadows (illumination flattening), and binarizes text.
- **GUI (`app/main.py`)**: A Streamlit application is implemented, allowing users to upload/capture images, preview the auto-detected boundary, manually adjust the crop corners, and export the document.
- **OCR (`src/pdf/export.py`)**: PyTesseract is integrated to generate fully text-searchable PDFs. (Requires local installation of Tesseract-OCR).
- **Evaluation (`evaluate.py` & `ocr_eval.py`)**: Comprehensive scripts exist to measure bounding box accuracy, execution time, and Character Error Rate (CER).

## 3. Project Structure
- `main.py` -> The CLI entry point for processing single images.
- `app/main.py` -> The Streamlit GUI entry point.
- `src/` -> Contains all core DIP logic, categorized by technique (`detection`, `enhancement`, `morphology`, `perspective`, `preprocessing`, `segmentation`, `pdf`).
- `docs/NEXT_STEPS.md` -> The living task queue and roadmap. **Always read this file first to know what needs to be done next.**
- `requirements.txt` -> Project dependencies.

## 4. Key Directives & Constraints for AI Agents
1. **No Deep Learning for Core Pipeline**: The core task is to demonstrate classical DIP. Do not replace the pipeline with neural networks (e.g., don't use YOLO for boundary detection). OCR (Tesseract) is the only exception, as it's an end-stage feature.
2. **Measurement over Guessing**: If you change an algorithm parameter (e.g., threshold block size, blur kernel), you MUST justify it by running the evaluation scripts (`evaluate.py` or `ocr_eval.py`) to prove it actually improves the metrics. Do not commit untested parameter tweaks.
3. **Keep `NEXT_STEPS.md` Updated**: Whenever you finish a task, move it to a "Recently Completed" section and ensure the next immediate tasks are clearly defined.
4. **Git Workflow**: Always verify that tests pass before pushing. Create specific feature branches if instructed by the user, and use conventional commits (e.g., `feat:`, `fix:`).

## 5. Getting Started
```bash
# Set up the environment
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run the CLI
py -3.12 main.py dataset/raw/sample.jpg

# Run the GUI
py -3.12 -m streamlit run app/main.py
```
