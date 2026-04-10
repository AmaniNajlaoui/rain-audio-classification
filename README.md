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
| `export/` | Example export (`inference_config.json`, `rain_cnn_best.pt`) |
| `processed_binary_4s/manifest.csv` | Clip paths and labels (paths are from the original machine) |
| `baseline.py`, `data_processing.py` | Older / auxiliary scripts (not required for the main notebook) |

## Run it locally

1. **Clone** this repository.

2. **Create a virtual environment** (recommended):
   ```bash
   cd rain_detection
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Data:** This project expects folders of `.wav` clips under `processed_binary_4s/rain/` and `processed_binary_4s/no_rain/`.  
   - `.wav` files are **not** stored in Git (too large).  
   - Use your own clips in the same layout, or copy an existing `processed_binary_4s` tree locally.

4. **Paths in the notebook:** Open `rain_detection.ipynb` and set `DATA_ROOT` (and any other `Path(...)`) for your machine, e.g.:
   ```python
   DATA_ROOT = Path("processed_binary_4s")  # relative to this folder
   ```
   Replace any absolute paths from another machine with paths that match your clone.

5. **GPU:** Training uses CUDA if available; CPU works but is slower.

6. Run all cells in order from the setup section onward.

## License

Licensed under the **MIT License**. See [LICENSE](LICENSE).
