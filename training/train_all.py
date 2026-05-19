#!/usr/bin/env python3
"""
PSX Model Training Script - Train all sector models with high accuracy
Run this before deploying dashboard to generate trained models.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboardcolab import (
    SECTOR_GROUPS, train_lstm_gru, load_history, add_features,
    MODEL_DIR, FORCE_RETRAIN
)
import pandas as pd
from datetime import datetime

def train_sector(sector_name, symbols):
    print(f"\n{'='*60}")
    print(f"TRAINING SECTOR: {sector_name}")
    print(f"Symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
    print(f"{'='*60}")

    all_results = []
    for sym in symbols:
        print(f"\n▶ Training {sym} ...")
        try:
            df = load_history(sym, period="5y")
            if df.empty or len(df) < 100:
                print(f"  ⚠ Insufficient data for {sym}, skipping")
                continue
            df_feat = add_features(df)
            res = train_lstm_gru(sym, df_feat, force_retrain=FORCE_RETRAIN)
            print(f"  ✅ {sym} | MAPE: {res['mape']:.2f}% | DirAcc: {res['dir_acc']:.1f}% | Incremental: {res.get('incremental', False)}")
            all_results.append({'symbol': sym, **res})
        except Exception as e:
            print(f"  ❌ Error training {sym}: {e}")

    avg_dir = sum(r['dir_acc'] for r in all_results) / len(all_results) if all_results else 0
    print(f"\n📊 Sector {sector_name} average directional accuracy: {avg_dir:.1f}%")
    return all_results

def main():
    print("🚀 PSX Multi-Sector LSTM+GRU Training Pipeline")
    print(f"Models will be saved to: {MODEL_DIR}")
    print(f"Start time: {datetime.now()}")

    all_sectors = {k: v for k, v in SECTOR_GROUPS.items() if v}  # skip empty KSE100
    all_sectors["KSE100"] = list(SECTOR_GROUPS["Banking"])[:5]  # sample for KSE100

    for sector, symbols in all_sectors.items():
        train_sector(sector, symbols)

    print("\n✅ All sector training complete. Models ready for dashboard deployment.")
    print(f"Total models in {MODEL_DIR}: {len(os.listdir(MODEL_DIR))}")

if __name__ == "__main__":
    main()
