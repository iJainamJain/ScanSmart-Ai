# Deploying SmartScan AI

The Streamlit app is the product face of the project: upload or take a photo,
adjust the corners if detection got them wrong, add pages, export a PDF. It
works in a phone browser and can be installed to the home screen.

Everything below is already committed. The only step that needs you is the
deploy itself, because it requires your GitHub login.

## 1. Run it locally

```bash
py -3.12 -m streamlit run app/main.py
```

Opens at http://localhost:8501. To reach it from your phone on the same
Wi-Fi, use the "Network URL" the command prints.

## 2. Deploy to Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **New app** → pick the `iJainamJain/ScanSmart-Ai` repo, branch `main`.
3. Set **Main file path** to `app/main.py`.
4. Deploy.

That is the whole process. The repository already contains what the platform
needs:

| file | why it is needed |
|------|------------------|
| `requirements.txt` | Runtime dependencies only. Uses `opencv-python-headless` — the standard build needs libGL, which the cloud image does not have, and the project makes no `cv2` GUI calls. Upper version bounds are deliberately loose: pinning below a major version (`numpy<2`, `pillow<11`) leaves no prebuilt wheel on the platform's current Python 3.14, so pip compiles from source and fails on missing system headers. Dev tooling lives in `requirements-dev.txt` so deploy builds stay fast. |
| `packages.txt` | System packages. Installs `tesseract-ocr`, the OCR binary; without it searchable-PDF export silently falls back to image-only. |
| `.streamlit/config.toml` | 30 MB upload cap, static file serving for the PWA assets. |

**Expect the first deploy to be slow.** The repository carries ~98 MB of
dataset photos, which the platform clones before building. It is a one-off
cost per deploy, not per visit. If it becomes annoying, move `dataset/raw/`
to a release asset or Git LFS — but do not delete it, the images are the
project's evidence base.

## 3. Install it to a phone home screen

Open the deployed URL on the phone, then:

- **Android / Chrome** — menu → *Add to Home screen*.
- **iOS / Safari** — Share → *Add to Home Screen*.

It gets an icon and opens without browser chrome, which is most of what
makes a web app feel native.

### What this is and is not

This is an installable web app, not a full Progressive Web App. There is no
service worker, so **it does not work offline** — Streamlit runs the pipeline
on the server and needs a live connection. Streamlit owns the page `<head>`,
so the manifest and meta tags are injected into the DOM instead; browsers
honour a manifest link found there, which is enough for the icon and the
standalone window.

A genuinely offline-capable version would mean reimplementing the pipeline in
the browser, or replacing Streamlit with a custom frontend over a
`src/pipeline.py` API. `src/` has no UI dependencies, so that path is open —
but it is a rewrite of the interface layer, not a setting.

## 4. Native mobile

Deliberately not built. It is out of the graded scope for this course project
(see the "Product vision" section of [proposal.md](proposal.md)), and a
native client plus a backend is a substantial piece of work whose failure
mode — a half-finished app — demos worse than the working web one. The
architecture does not block it: `src/pipeline.py` is already a clean service
boundary, so a FastAPI wrapper plus a Flutter or React Native client is the
route if it is ever wanted.
