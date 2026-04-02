# Rain vs. no-rain audio classification

Binary classification of **4-second audio clips** using **mel-spectrogram** inputs and a **PyTorch CNN**. Includes training curves, test metrics, confusion matrix, ROC / PR curves, and exported weights for inference.

**Stack:** Python · PyTorch · librosa · scikit-learn · matplotlib

## Results (this dataset / split)

| Metric | Value |
|--------|--------|
| Best validation accuracy | ~99.5% (epoch 36) |
| Test accuracy | ~100% |
| ROC-AUC | ~1.00 |
| Average precision | ~1.00 |

*Numbers come from the notebook outputs on the collected rain vs. non-rain clips. Real-world generalization depends on new microphones, backgrounds, and weather.*

## Repository layout

| Path | Purpose |
|------|--------|
| `rain_detection.ipynb` | Main notebook: EDA, preprocessing, model, evaluation, export |
| `export/` | Example export (`inference_config.json`, optional `rain_cnn_best.pt`) |
| `processed_binary_4s/manifest.csv` | Clip paths and labels (paths are from the original machine) |
| `baseline.py`, `data_processing.py` | Older / auxiliary scripts (not required for the main notebook) |

## Run it locally

1. **Clone** this repo (after you publish it on GitHub).

2. **Create a virtual environment** (recommended):
   ```bash
   cd rain_detection
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Data:** This project uses folders of `.wav` clips under `processed_binary_4s/rain/` and `processed_binary_4s/no_rain/`.  
   - `.wav` files are **not** stored in Git (too large).  
   - Prepare your own clips in the same layout, **or** copy your existing `processed_binary_4s` from your PC into the clone.

4. **Paths in the notebook:** Open `rain_detection.ipynb` and set `DATA_ROOT` (and any other `Path(...)`) to your machine, e.g.:
   ```python
   DATA_ROOT = Path("processed_binary_4s")  # relative to this folder
   ```
   Replace any old absolute paths like `/home/amani/phd/rain_detection/...` with paths that match your clone.

5. **GPU:** Training uses CUDA if available; CPU works but is slower.

6. Run all cells in order from the setup section onward.

## Publish to GitHub (for your portfolio)

1. Create an account on [github.com](https://github.com) if needed.

2. On GitHub: **New repository** → name e.g. `rain-audio-classification` → Public → **Create** (no need to add README there if this folder already has one).

3. On your PC, in this folder:
   ```bash
   cd /path/to/rain_detection
   git init
   git add README.md requirements.txt .gitignore rain_detection.ipynb export/
   git add processed_binary_4s/manifest.csv
   git add baseline.py data_processing.py rain_regression_experiments.ipynb rain_regression_paper.ipynb 2>/dev/null
   # Optional: include trained weights (~5 MB)
   git add export/rain_cnn_best.pt
   git commit -m "Portfolio: rain vs no-rain audio CNN"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```
   Replace `YOUR_USERNAME` and `YOUR_REPO` with yours.

4. If Git asks to log in, use a **Personal Access Token** (GitHub → Settings → Developer settings) as the password, or GitHub CLI (`gh auth login`).

## Add the project to Upwork

1. **Profile → Portfolio → Add** (wording may vary).

2. **Title example:** *Rain vs. non-rain detection from environmental audio (CNN + mel-spectrogram)*

3. **URL:** your GitHub repo link, e.g. `https://github.com/YOUR_USERNAME/YOUR_REPO`

4. **Description (you can paste and edit):**
   > Built a binary classifier on 4 s audio segments: rain vs. no rain. Converted waveforms to mel-spectrograms, trained a convolutional network in PyTorch with train/validation/test splits, and reported accuracy, confusion matrix, ROC-AUC, and precision–recall. Exported the best weights and inference settings for deployment. Stack: Python, PyTorch, librosa, scikit-learn.

5. **Screenshots:** Open the notebook, rerun the evaluation section, and save as PNG:
   - Training loss / accuracy curves  
   - Confusion matrix  
   - ROC or precision–recall plot  
   Upload these images in the Upwork portfolio item.

6. **Optional:** Record a 2-minute **Loom** walkthrough and paste the link in the description.

## License

Add a license (e.g. MIT) on GitHub if you want others to reuse the code.
