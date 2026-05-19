# PSX Professional Dashboard - Streamlit Deployment

Complete folder structure for deploying the PSX LSTM+GRU Multi-Sector Stock Dashboard on Streamlit.

## Folder Structure
```
.
├── app/
│   └── dashboard.py          # Main Streamlit app (unchanged interface)
├── training/
│   └── train_all.py          # Training script - trains all sector models with accuracy logging
├── models/                   # Saved models (.pth, .pkl) - populated after training
├── utils/                    # Shared utilities
├── data/                     # Optional cached data
├── docs/                     # Documentation
├── requirements.txt
└── README.md
```

## Quick Start (Local)

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train all models (recommended before first run for best accuracy):
   ```
   python training/train_all.py
   ```
   - Trains LSTM+GRU + ensemble models for Banking, Oil&Gas, Fertilizer, Cement, Technology, KSE100
   - Saves models to `models/` with validation accuracy > 75% directional accuracy target

3. Run Streamlit dashboard:
   ```
   streamlit run app/dashboard.py
   ```

## Deploy on Streamlit Cloud / Sharing

- Push this repo to GitHub
- Deploy via https://share.streamlit.io
- Ensure `models/` folder with trained models is committed (or use persistent storage)

## Training Notes
- `train_all.py` uses incremental training logic and reports per-sector MAPE & directional accuracy
- Models use LSTM+GRU hybrid with attention for high accuracy on PSX data
- Run training periodically for latest market data

Interface and logic kept identical to original Colab version.
