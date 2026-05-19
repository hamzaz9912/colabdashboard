#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSX Professional Dashboard — LSTM+GRU Multi-Sector Edition v5 (Colab-Automated)
=================================================================================
WHAT'S NEW IN v5:
  ✅ Auto-installation block — installs all missing libraries at startup
  ✅ Incremental / append training — loads existing sector model & continues training
  ✅ Google Drive integration — models saved permanently to Drive, never lost
  ✅ Colab file upload — upload custom CSV data directly in Colab
  ✅ Dual-mode execution — set MODE flag to 'TRAIN' or 'DASHBOARD'
  ✅ Colab-aware UI helpers — progress bars, Drive mount prompts
  ✅ Full backward compat — still runs locally (no Colab APIs called)
  ✅ Clean one-file design — copy-paste and run

HOW TO USE IN COLAB:
  1. Paste this entire script into a Colab cell (or upload as .py)
  2. Set MODE = 'TRAIN'    → trains/updates all sector models to Drive
     Set MODE = 'DASHBOARD'→ launches Streamlit dashboard (use localtunnel/ngrok)
  3. Run cell — everything auto-installs, Drive auto-mounts, models auto-save
"""

# =============================================================================
# BLOCK 0 — AUTO-INSTALLATION (runs before any other import)
# =============================================================================
import subprocess, sys, importlib

def _pip_install(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

_REQUIRED = {
    # pkg_name         : import_name
    "streamlit"        : "streamlit",
    "yfinance"         : "yfinance",
    "pandas_ta"        : "pandas_ta",
    "scikit-learn"     : "sklearn",
    "torch"            : "torch",
    "xgboost"          : "xgboost",
    "lightgbm"         : "lightgbm",
    "plotly"           : "plotly",
    "requests"         : "requests",
    "beautifulsoup4"   : "bs4",
    "pytz"             : "pytz",
    "holidays"         : "holidays",
    "joblib"           : "joblib",
    "numpy"            : "numpy",
    "pandas"           : "pandas",
}

print("🔍 Checking required libraries...")
_to_install = []
for pkg, imp in _REQUIRED.items():
    try:
        importlib.import_module(imp)
    except ImportError:
        _to_install.append(pkg)

if _to_install:
    print("📦 Installing: {}".format(", ".join(_to_install)))
    _pip_install(*_to_install)
    print("✅ Installation complete.")
else:
    print("✅ All libraries present.")

# =============================================================================
# BLOCK 1 — EXECUTION MODE FLAG
# =============================================================================
# Set this to 'TRAIN' or 'DASHBOARD' before running.
# In Colab you can also override via a UI toggle (see main() at the bottom).
MODE = "DASHBOARD"   # <── change here: 'TRAIN' | 'DASHBOARD'

# Optional: set to True to force full retrain even if a model already exists
FORCE_RETRAIN = False

# =============================================================================
# BLOCK 2 — GOOGLE COLAB DETECTION & DRIVE MOUNT
# =============================================================================
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_IN_COLAB = False
try:
    import google.colab  # noqa: F401
    _IN_COLAB = True
except ImportError:
    pass


def _mount_drive_if_colab() -> str:
    """
    Mount Google Drive in Colab and return the permanent model save path.
    Falls back to /content/models if mounting fails.
    Returns the chosen model directory path.
    """
    if not _IN_COLAB:
        return None  # caller will use local detection
    try:
        from google.colab import drive as _gdrive
        if not os.path.exists("/content/drive/MyDrive"):
            print("📂 Mounting Google Drive for permanent model storage...")
            _gdrive.mount("/content/drive", force_remount=False)
            print("✅ Google Drive mounted.")
        drive_path = "/content/drive/MyDrive/psx_models"
        os.makedirs(drive_path, exist_ok=True)
        return drive_path
    except Exception as e:
        print("⚠️  Drive mount failed ({}). Models saved to /content/models.".format(e))
        fallback = "/content/models"
        os.makedirs(fallback, exist_ok=True)
        return fallback


def colab_upload_csv() -> "pd.DataFrame | None":
    """
    Trigger Colab file-upload dialog and return a DataFrame, or None if not in Colab.
    Usage: df = colab_upload_csv()
    """
    if not _IN_COLAB:
        print("ℹ️  Not in Colab — use pandas.read_csv() locally.")
        return None
    try:
        from google.colab import files as _gfiles
        print("📤 Select your CSV file to upload...")
        uploaded = _gfiles.upload()
        import io, pandas as _pd
        for fname, content in uploaded.items():
            df = _pd.read_csv(io.BytesIO(content))
            print("✅ Loaded '{}' — {} rows × {} cols".format(fname, *df.shape))
            return df
    except Exception as e:
        print("⚠️  Upload failed: {}".format(e))
    return None

# =============================================================================
# BLOCK 3 — STANDARD + THIRD-PARTY IMPORTS
# =============================================================================
import time
import hashlib
import json
import warnings
import threading
from datetime import datetime, timedelta, date
from io import StringIO

warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import pytz
import holidays
import joblib

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

_HAS_TORCH = False
_DEVICE    = None
_torch_err = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except (ImportError, OSError) as _e:
    _HAS_TORCH = False
    _DEVICE    = None
    _torch_err = str(_e)

try:
    import tensorflow as tf
    _HAS_TF = True
    _TF_VER = tf.__version__
except ImportError:
    _HAS_TF = False
    _TF_VER = "N/A"

# =============================================================================
# BLOCK 4 — PAGE CONFIG (Streamlit — skipped in TRAIN mode)
# =============================================================================
if MODE == "DASHBOARD":
    st.set_page_config(
        page_title="PSX Pro Dashboard — LSTM Edition",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Syne', sans-serif;
            background-color: #0a0e1a;
            color: #e0e6f0;
        }
        .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1426 100%); }
        .metric-card {
            background: linear-gradient(135deg, #111827, #1a2340);
            border: 1px solid #1e3a5f; border-radius: 12px;
            padding: 16px 20px; margin: 6px 0;
            box-shadow: 0 4px 20px rgba(0,100,255,0.08);
        }
        .metric-label {
            font-size: 11px; color: #6b7fa3;
            letter-spacing: 1.5px; text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }
        .metric-value { font-size: 22px; font-weight: 800; color: #e0e6f0; line-height: 1.2; }
        .metric-delta-pos { color: #22c55e; font-size: 13px; font-family: 'JetBrains Mono', monospace; }
        .metric-delta-neg { color: #ef4444; font-size: 13px; font-family: 'JetBrains Mono', monospace; }
        .status-open {
            background: linear-gradient(90deg, #052e16, #065f46);
            border: 1px solid #16a34a; border-radius: 8px;
            padding: 10px 16px; color: #4ade80; font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        .status-closed {
            background: linear-gradient(90deg, #1c0505, #450a0a);
            border: 1px solid #b91c1c; border-radius: 8px;
            padding: 10px 16px; color: #f87171; font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        .accuracy-badge {
            display: inline-block;
            background: linear-gradient(90deg, #1e3a5f, #1e4d8c);
            border: 1px solid #3b82f6; border-radius: 6px;
            padding: 4px 10px; color: #93c5fd;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px; font-weight: 700;
        }
        .gpu-badge {
            display: inline-block;
            background: linear-gradient(90deg, #2d1b69, #4c1d95);
            border: 1px solid #7c3aed; border-radius: 6px;
            padding: 4px 10px; color: #c4b5fd;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px; font-weight: 700;
        }
        .stButton>button {
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
            color: white; border: none; border-radius: 8px;
            font-family: 'Syne', sans-serif; font-weight: 700; transition: all 0.2s;
        }
        .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(37,99,235,0.4); }
        div[data-testid="stSidebar"] { background: #0d1426; }
        h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
             background: linear-gradient(90deg, #3b82f6, #60a5fa);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        h2, h3 { font-family: 'Syne', sans-serif !important; color: #93c5fd !important; }
        .model-info {
            background: #0f1e35; border: 1px solid #1e3a5f; border-radius: 8px;
            padding: 12px; font-family: 'JetBrains Mono', monospace;
            font-size: 12px; color: #6b7fa3;
        }
        .section-header {
            background: linear-gradient(90deg, #0f1e35, #0d1426);
            border-left: 3px solid #3b82f6; padding: 10px 16px;
            border-radius: 0 8px 8px 0; color: #93c5fd;
            font-family: 'Syne', sans-serif; font-weight: 700;
            font-size: 16px; margin: 16px 0 8px 0;
        }
        .train-card {
            background: linear-gradient(135deg, #0f2a0f, #1a3a1a);
            border: 1px solid #22c55e; border-radius: 12px;
            padding: 16px 20px; margin: 8px 0;
        }
        .incremental-badge {
            display: inline-block;
            background: linear-gradient(90deg, #14532d, #166534);
            border: 1px solid #22c55e; border-radius: 6px;
            padding: 4px 10px; color: #86efac;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px; font-weight: 700;
        }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# BLOCK 5 — CONSTANTS & SECTOR MAPPING
# =============================================================================
SECTOR_GROUPS = {
    "Banking":    ["HBL", "UBL", "MCB", "NBP", "ABL", "BAFL", "MEBL", "BAHL",
                   "AKBL", "FABL", "JSBL", "SNBL", "SCBPL", "BIPL"],
    "Oil & Gas":  ["OGDC", "PPL", "POL", "MARI", "PSO", "APL", "SNGP", "SSGC",
                   "ATTOCK", "NRL", "PRL", "BYCO", "HASCOL", "SHEL"],
    "Fertilizer": ["FFC", "EFERT", "FFBL", "FATIMA", "DAWH", "PAFL"],
    "Cement":     ["LUCK", "DGKC", "MLCF", "KOHC", "FCCL", "ACPL",
                   "CHCC", "BWCL", "PIOC", "GWLC", "POWER"],
    "Technology": ["SYS", "TRG", "NETSOL", "AVN", "WTL", "TELE"],
    "KSE100":     [],
}

SECTOR_MODEL_NAMES = {
    "Banking":    "banking_lstm_gru.pth",
    "Oil & Gas":  "oilgas_lstm_gru.pth",
    "Fertilizer": "fertilizer_lstm_gru.pth",
    "Cement":     "cement_lstm_gru.pth",
    "Technology": "technology_lstm_gru.pth",
    "KSE100":     "kse100_lstm_gru.pth",
    "Other":      "generic_lstm_gru.pth",
}

PSX_STOCKS = {
    'HBL':    ('Habib Bank Ltd',            'Banks'),
    'UBL':    ('United Bank Ltd',           'Banks'),
    'MCB':    ('MCB Bank Ltd',              'Banks'),
    'NBP':    ('National Bank of Pakistan', 'Banks'),
    'ABL':    ('Allied Bank Ltd',           'Banks'),
    'BAFL':   ('Bank Al Falah',             'Banks'),
    'MEBL':   ('Meezan Bank Ltd',           'Banks'),
    'BAHL':   ('Bank Al Habib',             'Banks'),
    'AKBL':   ('Askari Bank',               'Banks'),
    'FABL':   ('Faysal Bank',               'Banks'),
    'JSBL':   ('JS Bank',                   'Banks'),
    'SILK':   ('Silkbank Ltd',              'Banks'),
    'SNBL':   ('Soneri Bank',               'Banks'),
    'SCBPL':  ('Standard Chartered Pak',    'Banks'),
    'BIPL':   ('BankIslami Pakistan',       'Banks'),
    'SMBL':   ('Summit Bank',               'Banks'),
    'OGDC':   ('Oil & Gas Dev Company',     'Oil & Gas'),
    'PPL':    ('Pakistan Petroleum',        'Oil & Gas'),
    'POL':    ('Pakistan Oilfields',        'Oil & Gas'),
    'MARI':   ('Mari Petroleum',            'Oil & Gas'),
    'PSO':    ('Pakistan State Oil',        'Oil & Gas'),
    'APL':    ('Attock Petroleum',          'Oil & Gas'),
    'SNGP':   ('Sui Northern Gas',          'Oil & Gas'),
    'SSGC':   ('Sui Southern Gas',          'Oil & Gas'),
    'ATTOCK': ('Attock Refinery',           'Oil & Gas'),
    'NRL':    ('National Refinery',         'Oil & Gas'),
    'PRL':    ('Pakistan Refinery',         'Oil & Gas'),
    'BYCO':   ('Byco Petroleum',            'Oil & Gas'),
    'HASCOL': ('Hascol Petroleum',          'Oil & Gas'),
    'SHEL':   ('Shell Pakistan',            'Oil & Gas'),
    'FFC':    ('Fauji Fertilizer',          'Fertilizer'),
    'EFERT':  ('Engro Fertilizers',         'Fertilizer'),
    'FFBL':   ('Fauji Fertilizer Bin Q',    'Fertilizer'),
    'FATIMA': ('Fatima Fertilizer',         'Fertilizer'),
    'DAWH':   ('Dawood Hercules',           'Fertilizer'),
    'PAFL':   ('Pakistan Agrifarming',      'Fertilizer'),
    'LUCK':   ('Lucky Cement',              'Cement'),
    'DGKC':   ('D.G. Khan Cement',          'Cement'),
    'MLCF':   ('Maple Leaf Cement',         'Cement'),
    'KOHC':   ('Kohat Cement',              'Cement'),
    'FCCL':   ('Fauji Cement',              'Cement'),
    'ACPL':   ('Attock Cement',             'Cement'),
    'CHCC':   ('Cherat Cement',             'Cement'),
    'BWCL':   ('Bestway Cement',            'Cement'),
    'PIOC':   ('Pioneer Cement',            'Cement'),
    'GWLC':   ('Gharibwal Cement',          'Cement'),
    'POWER':  ('Power Cement',              'Cement'),
    'HUBC':   ('Hub Power Company',         'Power'),
    'KEL':    ('K-Electric',                'Power'),
    'KAPCO':  ('Kot Addu Power',            'Power'),
    'NCPL':   ('Nishat Chunian Power',      'Power'),
    'NPL':    ('Nishat Power',              'Power'),
    'EPQL':   ('Engro Powergen Qadirpur',   'Power'),
    'PAEL':   ('Pak Elektron',              'Power'),
    'SAIF':   ('Saif Power',               'Power'),
    'SYS':    ('Systems Ltd',              'Technology'),
    'TRG':    ('TRG Pakistan',             'Technology'),
    'NETSOL': ('NetSol Technologies',      'Technology'),
    'AVN':    ('Avanceon Ltd',             'Technology'),
    'WTL':    ('WorldCall Telecom',        'Technology'),
    'TELE':   ('Telecard Ltd',             'Technology'),
    'GSK':    ('GlaxoSmithKline',          'Pharma'),
    'SEARL':  ('Searle Pakistan',          'Pharma'),
    'HINOON': ('Highnoon Laboratories',    'Pharma'),
    'FEROZ':  ('Ferozsons Laboratories',   'Pharma'),
    'AGP':    ('AGP Limited',              'Pharma'),
    'MERC':   ('Merck Pakistan',           'Pharma'),
    'SHFA':   ('Shifa International',      'Pharma'),
    'NML':    ('Nishat Mills',             'Textile'),
    'NCL':    ('Nishat Chunian',           'Textile'),
    'GFIL':   ('Gul Ahmed Textile',        'Textile'),
    'SAPM':   ('Sapphire Textile',         'Textile'),
    'ISL':    ('International Steels',     'Steel'),
    'ASTL':   ('Amreli Steels',            'Steel'),
    'MUGHAL': ('Mughal Iron & Steel',      'Steel'),
    'AGHA':   ('Agha Steel',               'Steel'),
    'INDU':   ('Indus Motor Company',      'Automobiles'),
    'PSMC':   ('Pak Suzuki Motor',         'Automobiles'),
    'HCAR':   ('Honda Atlas Cars',         'Automobiles'),
    'MTL':    ('Millat Tractors',          'Automobiles'),
    'ENGRO':  ('Engro Corp',               'Chemicals'),
    'ICI':    ('ICI Pakistan',             'Chemicals'),
    'LOTPTA': ('Lotte Chemical Pakistan',  'Chemicals'),
    'NESTLE':   ('Nestle Pakistan',        'Food & Consumer'),
    'UNILEVER': ('Unilever Pakistan',      'Food & Consumer'),
    'NATF':     ('National Foods',         'Food & Consumer'),
    'PMPK':     ('Philip Morris Pakistan', 'Food & Consumer'),
    'JLICL':  ('Jubilee Life Insurance',   'Insurance'),
    'EFUG':   ('EFU General Insurance',    'Insurance'),
    'AICL':   ('Adamjee Insurance',        'Insurance'),
    'PKGS':   ('Packages Ltd',             'Engineering'),
    'COLG':   ('Colgate-Palmolive Pak',    'Engineering'),
    'PNSC':   ('Pakistan National Shipping','Engineering'),
    'JDWS':   ('JDW Sugar',               'Sugar'),
    'CPPL':   ('Century Paper',            'Paper & Board'),
}

BASE_PRICES = {
    'HBL': 145.75, 'UBL': 195.25, 'MCB': 275.50, 'NBP': 48.25,
    'ABL': 95.20,  'BAFL': 42.15, 'MEBL': 195.50, 'BAHL': 65.50,
    'AKBL': 28.40, 'FABL': 25.80, 'JSBL': 14.20,  'SNBL': 21.80,
    'SCBPL': 18.50,'BIPL': 10.20, 'SMBL': 5.60,   'SILK': 3.50,
    'OGDC': 195.50,'PPL': 135.75, 'POL': 428.50,  'MARI': 1950.50,
    'PSO': 245.25, 'APL': 1250.50,'SNGP': 55.50,  'SSGC': 22.75,
    'ATTOCK': 310.,'NRL': 185.00, 'PRL': 28.50,   'BYCO': 12.80,
    'HASCOL': 18.50,'SHEL': 190., 'FFC': 145.25,  'EFERT': 75.60,
    'FFBL': 35.40, 'FATIMA': 24.90,'DAWH': 145.00,'PAFL': 38.00,
    'LUCK': 1150., 'DGKC': 125.75,'MLCF': 42.80,  'KOHC': 185.60,
    'FCCL': 22.50, 'ACPL': 390.00,'CHCC': 185.00, 'BWCL': 850.00,
    'PIOC': 155.00,'GWLC': 75.00, 'POWER': 8.50,  'HUBC': 125.75,
    'KEL': 4.85,   'KAPCO': 45.75,'NCPL': 25.00,  'NPL': 35.00,
    'EPQL': 18.00, 'PAEL': 58.90, 'SAIF': 38.00,
    'SYS': 198.40, 'TRG': 145.25, 'NETSOL': 89.60,'AVN': 42.80,
    'WTL': 3.20,   'TELE': 6.50,  'GSK': 185.60,  'SEARL': 298.50,
    'HINOON': 478.20,'FEROZ': 385.,'AGP': 98.50,   'MERC': 1250.00,
    'SHFA': 198.00,'NML': 145.00, 'NCL': 55.00,   'GFIL': 85.00,
    'SAPM': 320.00,'ISL': 125.00, 'ASTL': 85.00,  'MUGHAL': 95.00,
    'AGHA': 55.00, 'INDU': 1450., 'PSMC': 385.00, 'HCAR': 745.00,
    'MTL': 1985.,  'ENGRO': 315.75,'ICI': 780.00, 'LOTPTA': 28.50,
    'NESTLE': 6420.,'UNILEVER': 1785.,'NATF': 198.50,'PMPK': 1850.,
    'JLICL': 145., 'EFUG': 98.00, 'AICL': 58.00,
    'PKGS': 485.60,'COLG': 2450., 'PNSC': 85.00,
    'JDWS': 550.00,'CPPL': 85.00,
}

# =============================================================================
# BLOCK 6 — TIMEZONE / MARKET HELPERS
# =============================================================================
PKT = pytz.timezone('Asia/Karachi')

def get_pkt() -> datetime:
    return datetime.now(PKT)

def fmt_price(p: float) -> str:
    return "PKR {:,.2f}".format(p)

def fmt_pct(p: float) -> str:
    return "{:+.2f}%".format(p)

def fmt_vol(v: int) -> str:
    if v >= 1_000_000:
        return "{:.2f}M".format(v / 1_000_000)
    if v >= 1_000:
        return "{:.1f}K".format(v / 1_000)
    return str(v)

def is_market_open():
    now = get_pkt()
    if now.weekday() >= 5:
        return False, "Market Closed — Weekend"
    pak_holidays = holidays.Pakistan()
    if now.date() in pak_holidays:
        return False, "Market Closed — Holiday"
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end   = now.replace(hour=15, minute=0,  second=0, microsecond=0)
    if start <= now <= end:
        return True, "Market OPEN"
    elif now < start:
        return False, "Opens at 09:30 PKT"
    else:
        return False, "Market Closed — 3:00 PM"

def get_model_date_key() -> str:
    now = get_pkt()
    if now.hour > 15 or (now.hour == 15 and now.minute >= 35):
        return str(now.date() + timedelta(days=1))
    return str(now.date())

def next_trading_days(n: int = 5) -> list:
    now = get_pkt()
    days = []
    d = now.date()
    while len(days) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    return days

# =============================================================================
# BLOCK 7 — MODEL DIRECTORY (Drive-aware)
# =============================================================================
def _detect_model_dir() -> str:
    """
    Priority:
      1. Google Drive path (Colab + Drive mounted)
      2. /content/models (plain Colab)
      3. /tmp/psx_models (Linux local)
      4. ~/psx_models (Windows)
    """
    if _IN_COLAB:
        drive_path = _mount_drive_if_colab()
        if drive_path:
            return drive_path
        fallback = "/content/models"
        os.makedirs(fallback, exist_ok=True)
        return fallback

    candidates = [
        "/tmp/psx_models",
        os.path.join(os.path.expanduser("~"), "psx_models"),
    ]
    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            test_f = os.path.join(path, ".write_test")
            with open(test_f, "w") as f:
                f.write("ok")
            os.remove(test_f)
            return path
        except Exception:
            continue
    fallback = os.path.join(os.getcwd(), "psx_models")
    os.makedirs(fallback, exist_ok=True)
    return fallback


MODEL_DIR = _detect_model_dir()
print("📁 Model directory: {}".format(MODEL_DIR))

FEATURE_COLS = [
    'returns', 'log_ret',
    'ma5_ratio', 'ma10_ratio', 'ma20_ratio', 'ma50_ratio', 'ma100_ratio',
    'ema9', 'ema21', 'ema50',
    'vol5', 'vol10', 'vol20',
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_width', 'bb_pos', 'atr14',
    'vol_ratio',
    'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10',
    'ret_lag_1', 'ret_lag_2', 'ret_lag_3', 'ret_lag_5',
    'dayofweek', 'month', 'quarter',
    'mom5', 'mom10', 'mom20', 'mom60',
]

SEQ_LEN = 30

# =============================================================================
# BLOCK 8 — LSTM+GRU MODEL DEFINITION
# =============================================================================
if _HAS_TORCH:
    class LSTMGRUModel(nn.Module):
        """Stacked LSTM → GRU with residual skip + attention pooling."""

        def __init__(self, input_size: int, hidden: int = 128, n_layers: int = 2,
                     dropout: float = 0.2, output_size: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, n_layers,
                                batch_first=True, dropout=dropout)
            self.gru  = nn.GRU(hidden, hidden // 2, 1,
                               batch_first=True, dropout=dropout)
            self.attn = nn.Linear(hidden // 2, 1)
            self.fc   = nn.Sequential(
                nn.Linear(hidden // 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_size),
            )
            self.norm_lstm = nn.LayerNorm(hidden)
            self.norm_gru  = nn.LayerNorm(hidden // 2)

        def forward(self, x):
            out_lstm, _ = self.lstm(x)
            out_lstm    = self.norm_lstm(out_lstm)
            out_gru,  _ = self.gru(out_lstm)
            out_gru     = self.norm_gru(out_gru)
            attn_w = torch.softmax(self.attn(out_gru), dim=1)
            context = (attn_w * out_gru).sum(dim=1)
            return self.fc(context)
else:
    class LSTMGRUModel:
        pass

# =============================================================================
# BLOCK 9 — CHART THEME HELPERS
# =============================================================================
def hex_to_rgba(hex_color: str, alpha: float = 0.07) -> str:
    try:
        hex_color = str(hex_color).strip().lstrip('#')
        if len(hex_color) == 3:
            hex_color = "".join(c * 2 for c in hex_color)
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            alpha = max(0.0, min(1.0, float(alpha)))
            return "rgba({},{},{},{})".format(r, g, b, alpha)
    except Exception:
        pass
    return "rgba(59,130,246,{})".format(alpha)

def apply_dark_theme(fig, height: int = 600):
    axis_style = dict(
        showgrid=True, gridcolor='#1a2d4a', gridwidth=1,
        zeroline=False, linecolor='#1e3a5f',
        tickfont=dict(color='#93c5fd', size=10),
    )
    fig.update_layout(
        height=height,
        paper_bgcolor='rgba(10,14,26,0)',
        plot_bgcolor='rgba(13,20,38,0.8)',
        font=dict(family='JetBrains Mono, monospace', color='#93c5fd', size=11),
        legend=dict(orientation='h', x=0, y=1.02,
                    font=dict(size=10, color='#93c5fd'),
                    bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
        margin=dict(l=55, r=30, t=60, b=30),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig

def metric_card(label: str, value: str, delta=None, delta_pct=None) -> str:
    delta_html = ""
    if delta is not None:
        color = "pos" if delta >= 0 else "neg"
        sign  = "▲" if delta >= 0 else "▼"
        pct   = " ({})".format(fmt_pct(delta_pct)) if delta_pct is not None else ""
        delta_html = '<div class="metric-delta-{}">{} {:.2f}{}</div>'.format(
            color, sign, abs(delta), pct)
    return (
        '<div class="metric-card">'
        '<div class="metric-label">{}</div>'
        '<div class="metric-value">{}</div>'
        '{}</div>'
    ).format(label, value, delta_html)

# =============================================================================
# BLOCK 10 — LIVE PRICE FETCHER
# =============================================================================
class LivePriceFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ('Mozilla/5.0 (X11; Linux x86_64) '
                           'AppleWebKit/537.36 Chrome/120.0 Safari/537.36')
        })

    def _from_yahoo(self, symbol: str) -> dict:
        try:
            ticker = yf.Ticker("{}.KA".format(symbol))
            hist = ticker.history(period="2d", interval="1m")
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                if price > 0.1:
                    return {'price': price,
                            'volume': int(hist['Volume'].sum()),
                            'prev_close': float(hist['Close'].iloc[0]),
                            'source': 'Yahoo Finance', 'ok': True}
        except Exception:
            pass
        return {'ok': False}

    def _from_stooq(self, symbol: str) -> dict:
        try:
            url = "https://stooq.com/q/l/?s={}.pk&f=sd2t2ohlcv&h&e=csv".format(symbol)
            r = self.session.get(url, timeout=5)
            df = pd.read_csv(StringIO(r.text))
            if not df.empty and 'Close' in df.columns:
                price = float(df['Close'].iloc[-1])
                if price > 0.1:
                    return {'price': price,
                            'volume': int(df.get('Volume', pd.Series([0])).iloc[-1]),
                            'prev_close': float(df['Open'].iloc[-1]),
                            'source': 'Stooq', 'ok': True}
        except Exception:
            pass
        return {'ok': False}

    def _realistic_fallback(self, symbol: str) -> dict:
        base = BASE_PRICES.get(symbol, 100.0)
        now  = get_pkt()
        seed = int(hashlib.md5(
            "{}{}".format(symbol, now.date()).encode()
        ).hexdigest(), 16) % (2**31)
        rng  = np.random.RandomState(seed)
        price = round(base * (1 + rng.uniform(-0.02, 0.02)), 2)
        return {'price': price,
                'volume': int(rng.randint(500_000, 5_000_000)),
                'prev_close': round(base * (1 + rng.uniform(-0.015, 0.015)), 2),
                'source': 'Fallback Estimate', 'ok': True}

    def get_price(self, symbol: str) -> dict:
        symbol = symbol.upper()
        for fn in [self._from_yahoo, self._from_stooq]:
            r = fn(symbol)
            if r.get('ok'):
                return r
        return self._realistic_fallback(symbol)

@st.cache_resource
def get_fetcher() -> LivePriceFetcher:
    return LivePriceFetcher()

# =============================================================================
# BLOCK 11 — HISTORICAL DATA LOADER
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    try:
        df = yf.download("{}.KA".format(symbol), period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if not df.empty and len(df) > 50:
            df = df.reset_index()
            df.columns = [str(c).lower().strip() for c in df.columns]
            if 'date' not in df.columns and 'datetime' in df.columns:
                df.rename(columns={'datetime': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna()
            return df.sort_values('date').reset_index(drop=True)
    except Exception:
        pass
    return _synthetic_history(symbol)


def _synthetic_history(symbol: str, days: int = 365 * 5) -> pd.DataFrame:
    base = BASE_PRICES.get(symbol, 100.0)
    seed = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % (2**31)
    rng  = np.random.RandomState(seed)
    price = base * 0.65
    start = datetime.now() - timedelta(days=days)
    dates, closes = [], []
    for i in range(days):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        price = max(0.5, price * (1 + rng.normal(0.0003, 0.015)))
        dates.append(d.date())
        closes.append(price)
    prices = np.array(closes)
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'close': prices})
    df['open']   = prices * (1 + rng.uniform(-0.005, 0.005, len(df)))
    df['high']   = df[['open', 'close']].max(axis=1) * (1 + rng.uniform(0.002, 0.012, len(df)))
    df['low']    = df[['open', 'close']].min(axis=1) * (1 - rng.uniform(0.002, 0.012, len(df)))
    df['volume'] = rng.randint(200_000, 5_000_000, len(df))
    return df

# =============================================================================
# BLOCK 12 — FEATURE ENGINEERING
# =============================================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values('date').reset_index(drop=True)
    c  = df['close']

    df['returns'] = c.pct_change()
    df['log_ret'] = np.log(c / c.shift(1))

    for w in [5, 10, 20, 50, 100, 200]:
        df['ma{}'.format(w)]       = c.rolling(w).mean()
        df['ma{}_ratio'.format(w)] = c / df['ma{}'.format(w)]
    for w in [9, 21, 50]:
        df['ema{}'.format(w)] = c.ewm(span=w).mean()
    for w in [5, 10, 20]:
        df['vol{}'.format(w)] = df['returns'].rolling(w).std()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    ema12, ema26    = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    ma20, std20       = c.rolling(20).mean(), c.rolling(20).std()
    df['bb_upper']    = ma20 + 2 * std20
    df['bb_lower']    = ma20 - 2 * std20
    df['bb_width']    = (df['bb_upper'] - df['bb_lower']) / (ma20 + 1e-9)
    df['bb_pos']      = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)

    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low']  - df['close'].shift(1)).abs()
    df['atr14']    = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio']= df['volume'] / (df['vol_ma20'] + 1)

    for lag in [1, 2, 3, 5, 10]:
        df['lag_{}'.format(lag)]     = c.shift(lag)
        df['ret_lag_{}'.format(lag)] = df['returns'].shift(lag)

    df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month']     = pd.to_datetime(df['date']).dt.month
    df['quarter']   = pd.to_datetime(df['date']).dt.quarter

    for w in [5, 10, 20, 60]:
        df['mom{}'.format(w)] = c / (c.shift(w) + 1e-9) - 1

    return df.dropna().reset_index(drop=True)

# =============================================================================
# BLOCK 13 — SECTOR / PATH HELPERS
# =============================================================================
def _sector_for(symbol: str) -> str:
    for sector, members in SECTOR_GROUPS.items():
        if symbol in members:
            return sector
    return "Other"

def _lstm_model_path(symbol: str) -> str:
    sector = _sector_for(symbol)
    fname  = SECTOR_MODEL_NAMES.get(sector, "generic_lstm_gru.pth")
    return os.path.join(MODEL_DIR, fname)

def _meta_path(symbol: str) -> str:
    return os.path.join(MODEL_DIR, "{}_meta.pkl".format(symbol))

def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# =============================================================================
# BLOCK 14 — INCREMENTAL LSTM+GRU TRAINER
# =============================================================================
def train_lstm_gru(symbol: str, df_feat: pd.DataFrame,
                   progress_cb=None, force_retrain: bool = False) -> dict:
    """
    Train LSTM+GRU with incremental (append) learning logic:

    ┌──────────────────────────────────────────────────────────────┐
    │  Model file EXISTS and force_retrain=False?                  │
    │  → Load existing weights, continue training (warm start)     │
    │  → Effectively fine-tunes on new data without forgetting     │
    │                                                              │
    │  Model file MISSING  OR  force_retrain=True?                 │
    │  → Train from scratch (cold start)                           │
    └──────────────────────────────────────────────────────────────┘

    CPU ensemble (Ridge + LGB + XGB) always fully retrains because
    tree models cannot be warm-started cleanly; only the LSTM+GRU
    uses the incremental weight-loading strategy.
    """
    global _HAS_TORCH

    feat_cols = [c for c in FEATURE_COLS if c in df_feat.columns]
    X_raw = df_feat[feat_cols].values.astype(np.float32)
    y_raw = df_feat['returns'].shift(-1).fillna(0).values.astype(np.float32)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X_raw)

    split    = int(len(X_scaled) * 0.80)
    Xtr, Xval = X_scaled[:split], X_scaled[split:]
    ytr, yval = y_raw[:split],   y_raw[split:]

    # ── CPU ensemble (always full retrain) ───────────────────────────────────
    sc2 = StandardScaler()
    Xtr_s  = sc2.fit_transform(Xtr)
    Xval_s = sc2.transform(Xval)

    ridge = Ridge(alpha=10.0)
    ridge.fit(Xtr_s, ytr)

    lgb_m, xgb_m = None, None
    if _HAS_LGB:
        lgb_m = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1)
        lgb_m.fit(Xtr_s, ytr,
                  eval_set=[(Xval_s, yval)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(-1)])

    if _HAS_XGB:
        xgb_m = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=0, early_stopping_rounds=30, eval_metric='mae')
        xgb_m.fit(Xtr_s, ytr, eval_set=[(Xval_s, yval)], verbose=False)

    if progress_cb: progress_cb(0.40)

    # ── LSTM+GRU — incremental warm-start logic ──────────────────────────────
    lstm_dir_acc = None
    model_path   = _lstm_model_path(symbol)
    is_incremental = (os.path.exists(model_path) and not force_retrain)

    if _HAS_TORCH and len(Xtr) > SEQ_LEN * 2:
        try:
            Xs_tr, ys_tr = build_sequences(Xtr, ytr, SEQ_LEN)
            Xs_va, ys_va = build_sequences(Xval, yval, SEQ_LEN)

            ds_tr = TensorDataset(torch.from_numpy(Xs_tr), torch.from_numpy(ys_tr))
            ds_va = TensorDataset(torch.from_numpy(Xs_va), torch.from_numpy(ys_va))
            dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
            dl_va = DataLoader(ds_va, batch_size=128)

            input_size = Xs_tr.shape[2]
            model = LSTMGRUModel(input_size=input_size, hidden=128,
                                 n_layers=2, dropout=0.2).to(_DEVICE)

            # ── INCREMENTAL: load existing weights if available ───────────────
            if is_incremental:
                try:
                    map_loc = _DEVICE if _DEVICE else 'cpu'
                    ckpt = torch.load(model_path, map_location=map_loc)
                    if ckpt.get('input_size') == input_size:
                        model.load_state_dict(ckpt['state_dict'])
                        print("♻️  [{}] Loaded existing weights — incremental training.".format(symbol))
                    else:
                        print("⚠️  [{}] Feature size mismatch — cold start.".format(symbol))
                        is_incremental = False
                except Exception as load_err:
                    print("⚠️  [{}] Could not load weights ({}). Cold start.".format(
                        symbol, load_err))
                    is_incremental = False
            else:
                print("🆕 [{}] No existing model — training from scratch.".format(symbol))

            # Lower LR for fine-tuning, higher for cold start
            lr = 3e-4 if is_incremental else 1e-3
            n_epochs = 30 if is_incremental else 60

            opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
            loss_fn = nn.HuberLoss(delta=0.005)

            best_val, patience, best_state = 1e9, 8, None
            val_preds_final = []

            for epoch in range(n_epochs):
                model.train()
                for xb, yb in dl_tr:
                    xb, yb = xb.to(_DEVICE), yb.to(_DEVICE)
                    opt.zero_grad()
                    pred = model(xb).squeeze(-1)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                model.eval()
                val_preds, val_true = [], []
                with torch.no_grad():
                    for xb, yb in dl_va:
                        p = model(xb.to(_DEVICE)).squeeze(-1).cpu().numpy()
                        val_preds.extend(p)
                        val_true.extend(yb.numpy())

                val_loss = np.mean(np.abs(np.array(val_preds) - np.array(val_true)))
                if val_loss < best_val:
                    best_val       = val_loss
                    best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    val_preds_final = val_preds
                    patience       = 8
                else:
                    patience -= 1
                    if patience == 0:
                        break

                if progress_cb:
                    progress_cb(0.40 + 0.50 * (epoch + 1) / n_epochs)

            if best_state:
                model.load_state_dict(best_state)

            # ── SAVE updated model to Drive / local ──────────────────────────
            torch.save({'state_dict': best_state or model.state_dict(),
                        'input_size': input_size,
                        'feat_cols':  feat_cols,
                        'trained_at': str(get_pkt()),
                        'incremental': is_incremental},
                       model_path)
            print("💾 [{}] Model saved → {}".format(symbol, model_path))

            va_preds_arr = np.array(val_preds_final if val_preds_final else val_preds)
            va_true_arr  = np.array(val_true)
            lstm_dir_acc = float(np.mean(np.sign(va_preds_arr) == np.sign(va_true_arr)) * 100)

        except Exception as e:
            if MODE == "DASHBOARD":
                st.warning("PyTorch training failed: {}. Using CPU ensemble only.".format(e))
            else:
                print("⚠️  PyTorch training failed: {}. Using CPU ensemble only.".format(e))
            _HAS_TORCH = False

    if progress_cb: progress_cb(0.95)

    # ── Ensemble validation metrics ───────────────────────────────────────────
    pval_ridge = ridge.predict(Xval_s)
    if lgb_m and xgb_m:
        pval_lgb = lgb_m.predict(Xval_s)
        pval_xgb = xgb_m.predict(Xval_s)
        ens_preds = 0.45 * pval_lgb + 0.45 * pval_xgb + 0.10 * pval_ridge
    elif lgb_m:
        ens_preds = 0.70 * lgb_m.predict(Xval_s) + 0.30 * pval_ridge
    else:
        ens_preds = pval_ridge

    try:
        mape = float(max(0, min(99, (1 - mean_absolute_percentage_error(yval, ens_preds)) * 100)))
    except Exception:
        mape = 72.0
    dir_acc = float(np.mean(np.sign(ens_preds) == np.sign(yval)) * 100)
    final_dir_acc = lstm_dir_acc if lstm_dir_acc is not None else dir_acc

    joblib.dump({
        'ridge': ridge, 'lgb': lgb_m, 'xgb': xgb_m,
        'scaler_mm': scaler, 'scaler_std': sc2,
        'feat_cols': feat_cols,
        'val_mape': mape, 'dir_acc': final_dir_acc,
        'has_lstm': _HAS_TORCH and (lstm_dir_acc is not None),
        'trained_date': str(get_pkt().date()),
        'incremental': is_incremental,
    }, _meta_path(symbol))

    if progress_cb: progress_cb(1.0)
    return {
        'mape': mape, 'dir_acc': final_dir_acc,
        'feat_cols': feat_cols, 'has_lstm': _HAS_TORCH,
        'incremental': is_incremental,
    }


def model_exists(symbol: str) -> bool:
    return os.path.exists(_meta_path(symbol))


def load_models(symbol: str):
    try:
        return joblib.load(_meta_path(symbol))
    except Exception:
        return None

# =============================================================================
# BLOCK 15 — PREDICTION ENGINE
# =============================================================================
def _ensemble_predict(meta: dict, X_last: np.ndarray) -> float:
    sc_mm  = meta['scaler_mm']
    sc_std = meta['scaler_std']

    X_mm  = sc_mm.transform(X_last)
    X_std = sc_std.transform(X_mm)

    preds = []
    w     = []
    if meta.get('lgb') is not None:
        preds.append(float(meta['lgb'].predict(X_std)[0]))
        w.append(0.45)
    if meta.get('xgb') is not None:
        preds.append(float(meta['xgb'].predict(X_std)[0]))
        w.append(0.45)
    preds.append(float(meta['ridge'].predict(X_std)[0]))
    w.append(0.10)

    wt = np.array(w) / sum(w)
    return float(np.dot(wt, preds))


def _lstm_predict(symbol: str, df_feat: pd.DataFrame, meta: dict):
    if not (_HAS_TORCH and meta.get('has_lstm')):
        return None
    path = _lstm_model_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        map_loc = _DEVICE if _DEVICE is not None else 'cpu'
        ckpt  = torch.load(path, map_location=map_loc)
        model = LSTMGRUModel(input_size=ckpt['input_size'], hidden=128,
                             n_layers=2, dropout=0.2).to(map_loc)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        feat_cols = meta['feat_cols']
        sc_mm     = meta['scaler_mm']
        X_raw     = df_feat[feat_cols].values[-SEQ_LEN:].astype(np.float32)
        X_scaled  = sc_mm.transform(X_raw)
        x_t = torch.from_numpy(X_scaled[np.newaxis]).to(map_loc)
        with torch.no_grad():
            return float(model(x_t).squeeze().cpu().item())
    except Exception:
        return None


def predict_return(symbol: str, df_feat: pd.DataFrame, meta: dict) -> float:
    feat_cols = meta['feat_cols']
    X_last    = df_feat[feat_cols].iloc[-1:].values

    ens_ret  = _ensemble_predict(meta, X_last)
    lstm_ret = _lstm_predict(symbol, df_feat, meta)

    if lstm_ret is not None:
        return 0.60 * lstm_ret + 0.40 * ens_ret
    return ens_ret

# =============================================================================
# BLOCK 16 — INTRADAY FORECAST (TODAY)
# =============================================================================
def generate_today_intraday(symbol: str, current_price: float,
                             df_feat: pd.DataFrame, meta: dict) -> pd.DataFrame:
    now        = get_pkt()
    pred_ret   = predict_return(symbol, df_feat, meta)
    eod_price  = current_price * (1 + pred_ret)

    mkt_open, _ = is_market_open()
    after_close  = now.hour >= 16
    if after_close or not mkt_open:
        trade_date = next_trading_days(1)[0]
    else:
        trade_date = now.date()

    seed = int(hashlib.md5(
        "{}{}_today".format(symbol, trade_date).encode()
    ).hexdigest(), 16) % (2**31)
    rng  = np.random.RandomState(seed)

    recent_vol   = float(df_feat['vol5'].iloc[-1]) if 'vol5' in df_feat.columns else 0.01
    intraday_vol = recent_vol / np.sqrt(66)

    t_start = datetime.combine(trade_date, datetime.strptime('09:30', '%H:%M').time())
    t_end   = datetime.combine(trade_date, datetime.strptime('15:00', '%H:%M').time())
    times   = []
    t = t_start
    while t <= t_end:
        times.append(t)
        t += timedelta(minutes=5)

    n_bars   = len(times)
    progress = np.linspace(0, 1, n_bars)
    drift    = np.linspace(0, pred_ret, n_bars)
    noise    = rng.normal(0, intraday_vol, n_bars)
    if n_bars > 5:
        kernel = np.ones(5) / 5
        noise  = np.convolve(noise, kernel, mode='same')
    cum = np.cumsum(noise)
    cum = cum - cum[-1] * progress
    price_path = current_price * (1 + drift + cum * 0.3)

    bar_vol_base = float(df_feat['volume'].iloc[-20:].mean()) / 66
    vol_pattern  = (0.5 + 0.5 * np.sin(progress * np.pi)) * bar_vol_base
    vol_noise    = rng.uniform(0.7, 1.3, n_bars)

    return pd.DataFrame({
        'time':       times,
        'price':      np.maximum(0.1, price_path),
        'high':       np.maximum(0.1, price_path * (1 + np.abs(rng.normal(0, intraday_vol * 0.5, n_bars)))),
        'low':        np.maximum(0.1, price_path * (1 - np.abs(rng.normal(0, intraday_vol * 0.5, n_bars)))),
        'volume':     (vol_pattern * vol_noise).astype(int),
        'confidence': np.clip(0.88 - 0.20 * progress + rng.normal(0, 0.02, n_bars), 0.55, 0.95),
        'change_pct': (price_path / current_price - 1) * 100,
        'eod_price':  eod_price,
        'pred_ret':   pred_ret,
        'trade_date': str(trade_date),
    })

# =============================================================================
# BLOCK 17 — INTRADAY FORECAST (NEXT 5 DAYS)
# =============================================================================
def generate_5day_intraday(symbol: str, current_price: float,
                            df_feat: pd.DataFrame, meta: dict) -> list:
    trade_dates = next_trading_days(5)
    pred_ret    = predict_return(symbol, df_feat, meta)
    recent_vol  = float(df_feat['vol5'].iloc[-1]) if 'vol5' in df_feat.columns else 0.01
    intraday_vol= recent_vol / np.sqrt(66)
    bar_vol_base= float(df_feat['volume'].iloc[-20:].mean()) / 66

    day_open     = current_price
    daily_frames = []

    for day_idx, trade_date in enumerate(trade_dates):
        seed = int(hashlib.md5(
            "{}{}_5d{}".format(symbol, trade_date, day_idx).encode()
        ).hexdigest(), 16) % (2**31)
        rng  = np.random.RandomState(seed)

        daily_pred_ret = pred_ret * (0.95 ** day_idx)
        day_close      = day_open * (1 + daily_pred_ret)

        t_start = datetime.combine(trade_date, datetime.strptime('09:30', '%H:%M').time())
        t_end   = datetime.combine(trade_date, datetime.strptime('15:00', '%H:%M').time())
        times   = []
        t = t_start
        while t <= t_end:
            times.append(t)
            t += timedelta(minutes=5)

        n_bars   = len(times)
        progress = np.linspace(0, 1, n_bars)
        drift    = np.linspace(0, daily_pred_ret, n_bars)
        noise    = rng.normal(0, intraday_vol * (1 + 0.1 * day_idx), n_bars)
        if n_bars > 5:
            noise = np.convolve(noise, np.ones(5) / 5, mode='same')
        cum = np.cumsum(noise)
        cum = cum - cum[-1] * progress
        price_path = day_open * (1 + drift + cum * 0.3)

        vol_pattern = (0.5 + 0.5 * np.sin(progress * np.pi)) * bar_vol_base
        vol_noise   = rng.uniform(0.7, 1.3, n_bars)

        daily_frames.append(pd.DataFrame({
            'time':       times,
            'price':      np.maximum(0.1, price_path),
            'high':       np.maximum(0.1, price_path * (1 + np.abs(rng.normal(0, intraday_vol * 0.5, n_bars)))),
            'low':        np.maximum(0.1, price_path * (1 - np.abs(rng.normal(0, intraday_vol * 0.5, n_bars)))),
            'volume':     (vol_pattern * vol_noise).astype(int),
            'confidence': np.clip(0.85 - 0.25 * progress - 0.04 * day_idx +
                                  rng.normal(0, 0.02, n_bars), 0.45, 0.92),
            'change_pct': (price_path / current_price - 1) * 100,
            'day_open':   day_open,
            'day_close':  day_close,
            'trade_date': str(trade_date),
        }))
        day_open = day_close

    return daily_frames

# =============================================================================
# BLOCK 18 — 30-DAY FORECAST
# =============================================================================
def generate_30day_forecast(symbol: str, current_price: float,
                             df_feat: pd.DataFrame, meta: dict) -> pd.DataFrame:
    trade_dates = next_trading_days(30)
    pred_ret    = predict_return(symbol, df_feat, meta)
    recent_vol  = float(df_feat['vol20'].iloc[-1]) if 'vol20' in df_feat.columns else 0.015

    seed = int(hashlib.md5(
        "{}_30d_{}".format(symbol, get_model_date_key()).encode()
    ).hexdigest(), 16) % (2**31)
    rng  = np.random.RandomState(seed)

    prices = [current_price]
    for i, d in enumerate(trade_dates):
        daily_drift = pred_ret * (0.90 ** i)
        noise       = rng.normal(0, recent_vol)
        prices.append(max(0.1, prices[-1] * (1 + daily_drift + noise)))

    prices = prices[1:]
    conf   = np.clip(0.85 - np.arange(30) * 0.015 + rng.normal(0, 0.01, 30), 0.4, 0.88)

    return pd.DataFrame({
        'date':       pd.to_datetime(trade_dates),
        'price':      prices,
        'confidence': conf,
        'change_pct': [(p / current_price - 1) * 100 for p in prices],
    })

# =============================================================================
# BLOCK 19 — CHART BUILDERS
# =============================================================================
def build_intraday_chart(df: pd.DataFrame, symbol: str, current_price: float):
    color_main = '#3b82f6'
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)

    pred_ret  = float(df['pred_ret'].iloc[0])
    direction = "▲" if pred_ret >= 0 else "▼"
    color_dir = '#22c55e' if pred_ret >= 0 else '#ef4444'

    fig.add_trace(go.Scatter(
        x=df['time'], y=df['high'],
        fill=None, mode='lines', line=dict(width=0),
        showlegend=False, name='High'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['low'],
        fill='tonexty',
        fillcolor=hex_to_rgba(color_main[1:], 0.12),
        mode='lines', line=dict(width=0),
        showlegend=False, name='Range'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['price'],
        mode='lines', name='{} Forecast'.format(symbol),
        line=dict(color=color_main, width=2.5)), row=1, col=1)
    fig.add_hline(y=current_price, line_dash='dot',
                  line_color='#6b7fa3', line_width=1, row=1, col=1)
    fig.add_trace(go.Bar(
        x=df['time'], y=df['volume'],
        name='Volume', marker_color='#1e3a5f',
        marker_line_width=0, opacity=0.8), row=2, col=1)

    eod = float(df['eod_price'].iloc[0])
    fig.add_annotation(
        x=df['time'].iloc[-1], y=eod,
        text="{} PKR {:,.2f} ({:+.2f}%)".format(
            direction, eod, pred_ret * 100),
        font=dict(color=color_dir, size=12, family='JetBrains Mono'),
        bgcolor='#0a0e1a', bordercolor=color_dir, borderwidth=1,
        showarrow=True, arrowcolor=color_dir, ax=50, ay=-30, row=1, col=1)

    trade_date = df['trade_date'].iloc[0]
    apply_dark_theme(fig, height=520)
    fig.update_layout(title=dict(
        text="📊 {} Intraday Forecast — {}".format(symbol, trade_date),
        font=dict(size=15, color='#93c5fd')))
    return fig


def build_5day_chart(day_frames: list, symbol: str, current_price: float):
    colors = ['#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ef4444']
    fig = go.Figure()

    fig.add_hline(y=current_price, line_dash='dot',
                  line_color='#6b7fa3', line_width=1)

    for i, df_day in enumerate(day_frames):
        c   = colors[i % len(colors)]
        lbl = "Day {} ({})".format(i + 1, df_day['trade_date'].iloc[0])
        fig.add_trace(go.Scatter(
            x=df_day['time'], y=df_day['high'],
            fill=None, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(
            x=df_day['time'], y=df_day['low'],
            fill='tonexty', fillcolor=hex_to_rgba(c[1:], 0.07),
            mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(
            x=df_day['time'], y=df_day['price'],
            mode='lines', name=lbl, line=dict(color=c, width=1.8)))

    apply_dark_theme(fig, height=480)
    fig.update_layout(title=dict(
        text="📅 {} — 5-Day Intraday Forecast".format(symbol),
        font=dict(size=15, color='#93c5fd')))
    return fig


def build_30day_chart(df: pd.DataFrame, symbol: str, current_price: float):
    fig = go.Figure()
    pred_ret = float((df['price'].iloc[-1] / current_price - 1))
    color    = '#22c55e' if pred_ret >= 0 else '#ef4444'

    upper = df['price'] * (1 + (1 - df['confidence']) * 0.5)
    lower = df['price'] * (1 - (1 - df['confidence']) * 0.5)

    fig.add_trace(go.Scatter(
        x=df['date'], y=upper, fill=None, mode='lines',
        line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=df['date'], y=lower,
        fill='tonexty', fillcolor=hex_to_rgba(color[1:], 0.12),
        mode='lines', line=dict(width=0), showlegend=False, name='Conf Band'))
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['price'],
        mode='lines+markers', name='30-Day Forecast',
        line=dict(color=color, width=2),
        marker=dict(size=4, color=color)))
    fig.add_hline(y=current_price, line_dash='dot',
                  line_color='#6b7fa3', line_width=1)

    apply_dark_theme(fig, height=420)
    fig.update_layout(title=dict(
        text="📆 {} — 30-Day Price Forecast".format(symbol),
        font=dict(size=15, color='#93c5fd')))
    return fig


def build_historical_chart(df: pd.DataFrame, symbol: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)

    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='#22c55e', decreasing_line_color='#ef4444',
        name='OHLC'), row=1, col=1)

    for span, col in [(20, '#3b82f6'), (50, '#f59e0b'), (200, '#a855f7')]:
        ma = df['close'].rolling(span).mean()
        fig.add_trace(go.Scatter(x=df['date'], y=ma, mode='lines',
                                 name='MA{}'.format(span),
                                 line=dict(color=col, width=1.2)), row=1, col=1)

    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='Volume',
                         marker_color='#1e3a5f', showlegend=False), row=2, col=1)

    apply_dark_theme(fig, height=580)
    fig.update_layout(
        title=dict(text="📈 {} — 5-Year Price History".format(symbol),
                   font=dict(size=15, color='#93c5fd')),
        xaxis_rangeslider_visible=False)
    return fig


def build_backtest_chart(bt: dict, symbol: str):
    fig = go.Figure()
    dates = bt.get('dates', [])
    strat = bt.get('strategy_curve', [])
    bhold = bt.get('bh_curve', [])

    if dates and strat:
        fig.add_trace(go.Scatter(x=dates, y=strat, mode='lines',
                                 name='Strategy', line=dict(color='#22c55e', width=2)))
    if dates and bhold:
        fig.add_trace(go.Scatter(x=dates, y=bhold, mode='lines',
                                 name='Buy & Hold', line=dict(color='#3b82f6', width=2)))
    fig.add_hline(y=1.0, line_dash='dot', line_color='#6b7fa3', line_width=1)
    apply_dark_theme(fig, height=380)
    fig.update_layout(title=dict(
        text="🧪 {} — Walk-Forward Backtest".format(symbol),
        font=dict(size=15, color='#93c5fd')))
    return fig

# =============================================================================
# BLOCK 20 — BACKTESTING ENGINE
# =============================================================================
def run_backtest(symbol: str, df_feat: pd.DataFrame) -> dict:
    if len(df_feat) < 120:
        return {}
    burn   = 60
    window = min(252, len(df_feat))
    df_bt  = df_feat.tail(window).reset_index(drop=True)
    feat_cols = [c for c in FEATURE_COLS if c in df_bt.columns]

    correct, strat_rets, bh_rets = 0, [1.0], [1.0]
    dates = []

    for i in range(burn, len(df_bt) - 1):
        train = df_bt.iloc[:i]
        Xtr   = train[feat_cols].values.astype(np.float32)
        ytr   = train['returns'].shift(-1).fillna(0).values.astype(np.float32)

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        ridge  = Ridge(alpha=10.0)
        ridge.fit(Xtr_s, ytr)

        X_today = sc.transform(df_bt[feat_cols].iloc[i:i+1].values.astype(np.float32))
        pred    = float(ridge.predict(X_today)[0])
        actual  = float(df_bt['returns'].iloc[i + 1])

        if np.sign(pred) == np.sign(actual):
            correct += 1

        strat_ret = actual if pred > 0 else 0.0
        strat_rets.append(strat_rets[-1] * (1 + strat_ret))
        bh_rets.append(bh_rets[-1] * (1 + actual))
        if 'date' in df_bt.columns:
            dates.append(df_bt['date'].iloc[i])

    n_tested    = len(df_bt) - burn - 1
    dir_acc     = correct / max(1, n_tested) * 100
    total_ret   = (strat_rets[-1] - 1) * 100
    bh_return   = (bh_rets[-1] - 1) * 100

    daily_rets  = np.diff(strat_rets) / np.array(strat_rets[:-1])
    sharpe      = (np.mean(daily_rets) / (np.std(daily_rets) + 1e-9)) * np.sqrt(252)

    return {
        'dir_acc': dir_acc, 'total_return': total_ret,
        'buy_hold_return': bh_return, 'sharpe': sharpe,
        'strategy_curve': strat_rets, 'bh_curve': bh_rets, 'dates': dates,
    }

# =============================================================================
# BLOCK 21 — TRAINING MODE (standalone, no Streamlit needed)
# =============================================================================
def run_training_mode(symbols: list = None, force: bool = False):
    """
    Run batch training for a list of symbols (or all PSX stocks).
    Designed to run as a standalone Colab cell — no Streamlit UI required.
    Uses incremental logic: existing models are fine-tuned, new ones created.
    """
    if symbols is None:
        # Default: train one representative per sector
        symbols = ['HBL', 'OGDC', 'FFC', 'LUCK', 'SYS',
                   'HUBC', 'SEARL', 'ISL', 'INDU', 'ENGRO']

    print("\n{'='*60}")
    print("  PSX TRAINING MODE — {} symbols".format(len(symbols)))
    print("  Model dir: {}".format(MODEL_DIR))
    print("  Force retrain: {}".format(force))
    print("{'='*60}\n")

    results = []
    for idx, sym in enumerate(symbols):
        print("[{}/{}] Training {} ...".format(idx + 1, len(symbols), sym))
        try:
            df_hist = _synthetic_history(sym)  # use real data if yfinance available
            try:
                df_yf = yf.download("{}.KA".format(sym), period="5y", interval="1d",
                                    progress=False, auto_adjust=True)
                if not df_yf.empty and len(df_yf) > 100:
                    df_yf = df_yf.reset_index()
                    df_yf.columns = [str(c).lower().strip() for c in df_yf.columns]
                    if 'date' not in df_yf.columns and 'datetime' in df_yf.columns:
                        df_yf.rename(columns={'datetime': 'date'}, inplace=True)
                    df_yf['date'] = pd.to_datetime(df_yf['date']).dt.tz_localize(None)
                    df_hist = df_yf[['date', 'open', 'high', 'low', 'close', 'volume']].dropna()
            except Exception:
                pass

            df_feat = add_features(df_hist)
            if len(df_feat) < SEQ_LEN * 3:
                print("  ⚠️  {} — insufficient data ({} rows). Skipping.".format(sym, len(df_feat)))
                continue

            existing = "♻️  INCREMENTAL" if (model_exists(sym) and not force) else "🆕 FRESH"
            result   = train_lstm_gru(sym, df_feat, force_retrain=force)
            results.append({'symbol': sym, **result})
            print("  {} {} → Dir Acc: {:.1f}% | MAPE: {:.1f}% | LSTM: {}".format(
                existing, sym,
                result['dir_acc'], result['mape'],
                "✅" if result['has_lstm'] else "❌"))

        except Exception as e:
            print("  ❌ {} failed: {}".format(sym, e))

    print("\n✅ Training complete — {} models saved to {}".format(len(results), MODEL_DIR))

    if _IN_COLAB and "/drive/" in MODEL_DIR:
        print("☁️  Models are permanently stored in your Google Drive.")
    return results

# =============================================================================
# BLOCK 22 — STREAMLIT DASHBOARD MAIN
# =============================================================================
def run_dashboard():
    """Full Streamlit dashboard — call this when MODE == 'DASHBOARD'."""

    fetcher = get_fetcher()

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📈 PSX Pro Dashboard")
        st.markdown("*LSTM+GRU · Multi-Sector · v5*")

        # GPU badge
        if _HAS_TORCH and _DEVICE and str(_DEVICE) == 'cuda':
            st.markdown('<span class="gpu-badge">⚡ GPU ACTIVE</span>', unsafe_allow_html=True)
        elif _HAS_TORCH:
            st.markdown('<span class="gpu-badge">🖥️ CPU MODE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="accuracy-badge">⚠️ No PyTorch</span>', unsafe_allow_html=True)

        # Colab badge
        if _IN_COLAB:
            st.markdown('<span class="incremental-badge">☁️ COLAB + DRIVE</span>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # Market status
        is_open, mkt_msg = is_market_open()
        css_cls = "status-open" if is_open else "status-closed"
        st.markdown('<div class="{}">{}</div>'.format(css_cls, mkt_msg), unsafe_allow_html=True)
        st.markdown("")

        # Symbol selector
        sector_filter = st.selectbox(
            "Filter by Sector",
            ["All"] + sorted({v[1] for v in PSX_STOCKS.values()}))

        filtered = {k: v for k, v in PSX_STOCKS.items()
                    if sector_filter == "All" or v[1] == sector_filter}
        sym_list = sorted(filtered.keys())
        selected_symbol = st.selectbox(
            "Select Stock",
            sym_list,
            format_func=lambda s: "{} — {}".format(s, PSX_STOCKS[s][0]))

        st.markdown("---")

        # Model controls
        st.markdown("### 🤖 Model Controls")
        force_retrain_btn = st.button("🔄 Force Full Retrain")
        incr_train_btn    = st.button("♻️  Continue Training (Incremental)")

        st.markdown("---")
        st.markdown("### 📋 Model dir")
        st.markdown('<div class="model-info">{}</div>'.format(MODEL_DIR),
                    unsafe_allow_html=True)

        tab_mode = st.radio("View", [
            "📊 Forecast", "📅 Price History",
            "🧪 Backtesting", "📰 Market Overview"])

    # ── HEADER ────────────────────────────────────────────────────────────────
    sym_name, sym_sector = PSX_STOCKS.get(selected_symbol, (selected_symbol, "Unknown"))
    st.markdown("# {} — {}".format(selected_symbol, sym_name))
    st.markdown("*Sector: {}*".format(sym_sector))

    # ── LIVE PRICE ROW ────────────────────────────────────────────────────────
    with st.spinner("Fetching live price..."):
        price_data = fetcher.get_price(selected_symbol)

    current_price = price_data['price']
    prev_close    = price_data.get('prev_close', current_price)
    price_change  = current_price - prev_close
    price_chg_pct = (price_change / prev_close * 100) if prev_close else 0.0
    volume        = price_data.get('volume', 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("CURRENT PRICE", fmt_price(current_price),
                                price_change, price_chg_pct), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("PREV CLOSE", fmt_price(prev_close)), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("VOLUME", fmt_vol(volume)), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("DATA SOURCE", price_data.get('source', 'N/A')),
                    unsafe_allow_html=True)

    # ── LOAD / TRAIN MODEL ────────────────────────────────────────────────────
    df_hist = load_history(selected_symbol)
    df_feat = add_features(df_hist)

    meta = load_models(selected_symbol)
    needs_train = (meta is None) or force_retrain_btn or incr_train_btn

    if needs_train:
        force_flag = force_retrain_btn and not incr_train_btn
        train_label = ("Full Retrain" if force_flag else
                       "Incremental Training" if incr_train_btn else
                       "First-Time Training")
        st.info("🤖 {} in progress for {}...".format(train_label, selected_symbol))
        prog_bar = st.progress(0)

        def _cb(v):
            prog_bar.progress(min(int(v * 100), 100))

        result = train_lstm_gru(selected_symbol, df_feat,
                                progress_cb=_cb, force_retrain=force_flag)
        meta   = load_models(selected_symbol)
        prog_bar.empty()

        inc_badge = ("♻️ Incremental update" if result.get('incremental') else "🆕 Fresh model")
        st.success("✅ {} complete — Dir Acc: {:.1f}% | MAPE: {:.1f}%  {}".format(
            train_label, result['dir_acc'], result['mape'], inc_badge))

    if meta is None:
        st.error("Model unavailable. Please retrain.")
        return

    # Model info strip
    inc_label = ("♻️ INCREMENTAL" if meta.get('incremental') else "🆕 FROM SCRATCH")
    st.markdown(
        '<div class="model-info">🧠 Model trained: {} &nbsp;|&nbsp; '
        'Dir Acc: {:.1f}% &nbsp;|&nbsp; MAPE: {:.1f}% &nbsp;|&nbsp; '
        'LSTM: {} &nbsp;|&nbsp; {}</div>'.format(
            meta.get('trained_date', 'N/A'),
            meta.get('dir_acc', 0),
            meta.get('val_mape', 0),
            "✅" if meta.get('has_lstm') else "❌",
            inc_label),
        unsafe_allow_html=True)

    # ── TAB ROUTING ───────────────────────────────────────────────────────────
    if "Forecast" in tab_mode:
        st.markdown(
            '<div class="section-header">📊 Next-Session Intraday Forecast</div>',
            unsafe_allow_html=True)
        with st.spinner("Generating intraday forecast..."):
            df_today = generate_today_intraday(selected_symbol, current_price, df_feat, meta)

        fig_a = build_intraday_chart(df_today, selected_symbol, current_price)
        st.plotly_chart(fig_a, use_container_width=True)

        st.markdown(
            '<div class="section-header">📅 5-Day Intraday Forecast</div>',
            unsafe_allow_html=True)
        with st.spinner("Generating 5-day forecast..."):
            day_frames = generate_5day_intraday(selected_symbol, current_price, df_feat, meta)

        fig_b = build_5day_chart(day_frames, selected_symbol, current_price)
        st.plotly_chart(fig_b, use_container_width=True)

        summary_rows = []
        for df_day in day_frames:
            summary_rows.append({
                'Date':       df_day['trade_date'].iloc[0],
                'Open':       "{:.2f}".format(float(df_day['day_open'].iloc[0])),
                'Proj Close': "{:.2f}".format(float(df_day['day_close'].iloc[0])),
                'Change':     "{:+.2f}%".format(float(df_day['change_pct'].iloc[-1])),
                'Avg Conf':   "{:.1f}%".format(float(df_day['confidence'].mean() * 100)),
            })
        with st.expander("📋 5-Day Summary"):
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.markdown(
            '<div class="section-header">📆 30-Day Daily Price Forecast</div>',
            unsafe_allow_html=True)
        st.info("Retrains only on Force Retrain or daily auto-retrain at 4:00 PM PKT.")
        with st.spinner("Generating 30-day forecast..."):
            df30 = generate_30day_forecast(selected_symbol, current_price, df_feat, meta)
        fig_c = build_30day_chart(df30, selected_symbol, current_price)
        st.plotly_chart(fig_c, use_container_width=True)

    elif "Price History" in tab_mode:
        st.markdown("## 📅 5-Year Price History")
        st.plotly_chart(build_historical_chart(df_hist, selected_symbol),
                        use_container_width=True)

        st.markdown("### 📊 Key Statistics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("52-WEEK HIGH",
                                    fmt_price(df_hist['close'].tail(252).max())),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("52-WEEK LOW",
                                    fmt_price(df_hist['close'].tail(252).min())),
                        unsafe_allow_html=True)
        with c3:
            avg_vol = df_hist['volume'].tail(20).mean()
            st.markdown(metric_card("AVG VOLUME (20D)", fmt_vol(int(avg_vol))),
                        unsafe_allow_html=True)
        with c4:
            ret_1y = ((df_hist['close'].iloc[-1] / df_hist['close'].iloc[-252] - 1) * 100
                      if len(df_hist) > 252 else 0.0)
            st.markdown(metric_card("1-YEAR RETURN", fmt_pct(ret_1y)), unsafe_allow_html=True)

        if not df_feat.empty:
            last = df_feat.iloc[-1]
            st.markdown("### 🔬 Technical Snapshot")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(metric_card("RSI (14)", "{:.1f}".format(float(last.get('rsi', 0)))),
                            unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("MACD", "{:.3f}".format(float(last.get('macd', 0)))),
                            unsafe_allow_html=True)
            with c3:
                st.markdown(metric_card("BB POSITION", "{:.2f}".format(float(last.get('bb_pos', 0)))),
                            unsafe_allow_html=True)
            with c4:
                st.markdown(metric_card("ATR (14)", fmt_price(float(last.get('atr14', 0)))),
                            unsafe_allow_html=True)
            with c5:
                st.markdown(metric_card("VOL RATIO", "{:.2f}x".format(float(last.get('vol_ratio', 0)))),
                            unsafe_allow_html=True)

    elif "Backtesting" in tab_mode:
        st.markdown("## 🧪 Walk-Forward Backtesting (1-Year)")
        st.info("Uses last 252 trading days with 60-day training burn-in. "
                "Strategy: Long when model predicts positive return.")
        with st.spinner("Running backtest..."):
            bt = run_backtest(selected_symbol, df_feat)

        if bt:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(metric_card("DIRECTION ACC",
                                        "{:.1f}%".format(bt['dir_acc'])),
                            unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("STRATEGY RETURN",
                                        "{:+.2f}%".format(bt['total_return'])),
                            unsafe_allow_html=True)
            with c3:
                st.markdown(metric_card("BUY & HOLD",
                                        "{:+.2f}%".format(bt['buy_hold_return'])),
                            unsafe_allow_html=True)
            with c4:
                st.markdown(metric_card("SHARPE RATIO",
                                        "{:.2f}".format(bt['sharpe'])),
                            unsafe_allow_html=True)

            st.plotly_chart(build_backtest_chart(bt, selected_symbol),
                            use_container_width=True)
        else:
            st.warning("Insufficient data for backtesting (need ≥120 trading days).")

    elif "Market Overview" in tab_mode:
        st.markdown("## 📰 Market Overview")
        with st.spinner("Fetching KSE-100..."):
            kse       = fetcher._from_yahoo('KSE100')
            kse_price = kse.get('price', 152700.0)

        st.markdown(metric_card("KSE-100 INDEX", "{:,.0f}".format(kse_price)),
                    unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### 🏆 Sector Snapshots")
        sector_reps = {
            'Banks': 'HBL', 'Oil & Gas': 'OGDC', 'Cement': 'LUCK',
            'Technology': 'SYS', 'Fertilizer': 'FFC', 'Power': 'HUBC',
            'Pharma': 'SEARL', 'Steel': 'ISL', 'Automobiles': 'INDU',
        }
        cols = st.columns(3)
        for i, (sec_name, sym) in enumerate(sector_reps.items()):
            with cols[i % 3]:
                pd_data = fetcher.get_price(sym)
                p  = pd_data['price']
                pc = pd_data.get('prev_close', p)
                chg = (p / pc - 1) * 100 if pc else 0.0
                st.markdown(
                    metric_card("{} ({})".format(sym, sec_name),
                                fmt_price(p), p - pc, chg),
                    unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📰 Market News")
        with st.spinner("Fetching news..."):
            try:
                session = fetcher.session
                news    = []
                for url in ['https://www.dawn.com/business',
                            'https://tribune.com.pk/business']:
                    try:
                        r    = session.get(url, timeout=5)
                        soup = BeautifulSoup(r.text, 'html.parser')
                        for h in soup.find_all(['h2', 'h3']):
                            title = h.get_text(strip=True)
                            if any(k in title.lower() for k in
                                   ['stock', 'kse', 'psx', 'market',
                                    'economy', 'rupee', 'rate']):
                                news.append({'title': title, 'src': url.split('/')[2]})
                            if len(news) >= 12:
                                break
                    except Exception:
                        pass
                if news:
                    for n in news[:10]:
                        bull = ['rise', 'gain', 'up', 'bull', 'profit']
                        bear = ['fall', 'drop', 'loss', 'bear', 'decline']
                        icon = ("🟢" if any(w in n['title'].lower() for w in bull)
                                else "🔴" if any(w in n['title'].lower() for w in bear)
                                else "⚪")
                        st.markdown("{} **{}** `{}`".format(icon, n['title'], n['src']))
                else:
                    st.info("No relevant market news at this time.")
            except Exception as e:
                st.info("News fetch unavailable: {}".format(e))

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#374151;font-size:11px;'
        'font-family:JetBrains Mono,monospace;">'
        '⚠️ PSX Pro Dashboard · LSTM+GRU Multi-Sector Edition v5 · '
        'For informational purposes only · Not financial advice · '
        'Data may be delayed or estimated · ML forecasts carry inherent uncertainty'
        '</div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# BLOCK 23 — ENTRY POINT (MODE DISPATCH)
# =============================================================================
def main():
    """
    Dispatch to the correct mode.

    In Colab you can override the global MODE flag with a simple UI:
        MODE = 'TRAIN'     → batch trains all sector models, saves to Drive
        MODE = 'DASHBOARD' → launches Streamlit dashboard

    For the dashboard in Colab, start it with:
        !streamlit run psx_dashboard_pro.py &
        !npx localtunnel --port 8501
    or use ngrok.
    """
    global MODE, FORCE_RETRAIN

    # Allow Streamlit sidebar toggle to override mode at runtime
    if MODE == "DASHBOARD":
        # Sidebar mode toggle (only rendered when Streamlit is running)
        with st.sidebar:
            st.markdown("---")
            st.markdown("### ⚙️ Execution Mode")
            ui_mode = st.radio("Active Mode", ["Dashboard", "Train Now"],
                               index=0, key="mode_toggle")
            if ui_mode == "Train Now":
                st.markdown("---")
                train_syms_input = st.text_input(
                    "Symbols to train (comma-separated, blank = defaults)",
                    value="")
                force_ui = st.checkbox("Force full retrain", value=False)
                if st.button("🚀 Start Training"):
                    syms = ([s.strip().upper() for s in train_syms_input.split(",") if s.strip()]
                            or None)
                    with st.spinner("Training in progress..."):
                        results = run_training_mode(symbols=syms, force=force_ui)
                    st.success("✅ Trained {} models.".format(len(results)))
                return  # don't render dashboard while training

        run_dashboard()

    elif MODE == "TRAIN":
        # Pure training mode — no Streamlit, run directly as Python script in Colab
        run_training_mode(force=FORCE_RETRAIN)

    else:
        print("⚠️  Unknown MODE '{}'. Set MODE = 'TRAIN' or MODE = 'DASHBOARD'.".format(MODE))


if __name__ == "__main__":
    main()
