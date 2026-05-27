# 🏭 Wilmaru Plant Intelligence
### Full-Stack Multi-Agent AI System for Palm Oil Manufacturing Operations

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6B35?style=flat)](https://langchain-ai.github.io/langgraph/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io)
[![Cost](https://img.shields.io/badge/Biaya%20Bulanan-Rp%200-brightgreen?style=flat)]()

---

> **"Apa yang terjadi jika seluruh operasional pabrik CPO dimonitor dan dianalisis oleh AI — 24 jam, 7 hari seminggu?"**

Proyek ini menjawab pertanyaan itu dengan membangun sistem **9 AI agents** yang secara bersamaan memantau gudang, memprediksi kerusakan mesin, dan mengontrol kualitas produksi — lalu menggabungkan ketiganya menjadi satu executive insight.

---

## 🎬 Demo Cepat

```bash
python main.py --setup   # generate data simulasi (perlu dijalankan sekali)
python main.py --demo    # lihat semua agents bekerja
streamlit run dashboard/app.py  # buka dashboard interaktif
```

Contoh output dari Plant AI:

```
❓ Query: "Jika CENTRIFUGE-01 rusak besok, apa dampaknya?"

🤖 Plant Intelligence:

JAWABAN LANGSUNG:
CENTRIFUGE-01 memiliki RUL 31 jam — failure dalam 1-2 hari sangat mungkin.
Spare part yang dibutuhkan (BEARING-6205 x2, SEAL-PUMP-007 x1) TERSEDIA di warehouse.
Estimasi production loss jika tidak ditangani: 13.4 MT output dalam 8 jam downtime.

TEMUAN KRITIS:
• Health score mesin 38% — vibrasi 5.8 mm/s (2.8x normal)
• Korelasi vibrasi-to-reject rate: 0.71 (tinggi) — quality sudah terpengaruh
• Stock BEARING-6205: 142 pcs (cukup untuk 2x maintenance)
• LINE-A-CPO reject rate naik +2.1% dalam 3 hari terakhir

REKOMENDASI AKSI:
1. HARI INI: Jadwalkan maintenance darurat shift pagi besok (07:00-15:00)
2. SEKARANG: Ambil spare parts dari warehouse (Lokasi: Rak B-12)
3. BESOK: Alihkan sementara sebagian beban ke CENTRIFUGE-02
4. MINGGU INI: Audit PRESS-02 — RUL 78 jam, kondisi serupa
```

---

## 🏗️ Arsitektur: 9 Agents dalam 3 Modul

```
                    ┌─────────────────────────────────┐
                    │   🧠 Plant Orchestrator Agent    │
                    │     (LangGraph State Machine)    │
                    └──────────┬──────────┬───────────┘
                               │          │
           ┌───────────────────┼──────────┼───────────────────┐
           ▼                   ▼                              ▼
  ┌─────────────────┐ ┌─────────────────┐           ┌─────────────────┐
  │  📦 WAREHOUSE   │ │  🔧 MAINTENANCE  │           │  🎯 QUALITY     │
  │                 │ │                 │           │                 │
  │ Demand Forecast │ │ Sensor Monitor  │           │ Anomaly Detect  │
  │ Reorder Agent   │ │ RUL Predictor   │           │ Root Cause      │
  │ Expiry Risk     │ │ Schedule Agent  │           │ Yield Forecast  │
  └────────┬────────┘ └────────┬────────┘           └────────┬────────┘
           └──────────────────►│◄──────────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │  🔗 Cross-Domain    │
                    │  Insight Engine     │
                    │  (yang buat WOW)    │
                    └─────────────────────┘
```

### Cross-Domain: Yang Membuat Sistem Ini Unik

Kebanyakan sistem memantau satu domain saja. Sistem ini menghubungkan ketiga domain:

| Pertanyaan | Modul yang Terlibat |
|---|---|
| "Mesin rusak → spare part cukup?" | Maintenance + Warehouse |
| "Vibration tinggi → quality turun?" | Maintenance + Quality |
| "Jika downtime 8 jam → berapa MT loss?" | Maintenance + Quality |
| "Reorder spare part sebelum maintenance?" | Warehouse + Maintenance |

---

## 🤖 Detail 9 Agents

### Modul 1: Warehouse AI (3 agents)

| Agent | Input | Output | Algorithm |
|---|---|---|---|
| **Demand Forecast** | 90 hari historical | 30-day forecast per SKU | Prophet / Exp Smoothing |
| **Reorder Agent** | Stock + demand forecast | PO recommendation + EOQ | Economic Order Quantity |
| **Expiry Risk** | Stock + shelf life | Alert item kritis | Rule-based + threshold |

### Modul 2: Predictive Maintenance AI (3 agents)

| Agent | Input | Output | Algorithm |
|---|---|---|---|
| **Sensor Monitor** | Temperature, vibration, RPM, current | Anomaly per mesin | Isolation Forest |
| **RUL Predictor** | Time series sensor (90 hari) | Remaining Useful Life (hours) | Gradient Boosting + rolling features |
| **Schedule Agent** | RUL predictions + spare part | Jadwal maintenance optimal | Priority queue + safety buffer |

### Modul 3: Production Quality AI (3 agents)

| Agent | Input | Output | Algorithm |
|---|---|---|---|
| **Anomaly Detection** | FFA%, moisture%, color, reject rate | Batch anomaly flag | Isolation Forest + SPC rules |
| **Root Cause** | Sensor logs × quality params | Korelasi mesin → quality | Pearson correlation analysis |
| **Yield Forecast** | OEE, runtime, reject rate | 7-day output forecast | Ridge Regression |

---

## 📊 Dataset Simulasi Realistis

Semua dataset dibuat dengan `generate_data.py` berdasarkan karakteristik industri nyata:

### `machine_sensor_logs.csv` (32.760 baris)
7 mesin × 90 hari × 6 readings/hari. Termasuk degradation pattern yang mendekati failure.

```
timestamp           machine_id          temp_c  vib_mm_s  health_score  rul_hours  status
2024-10-01 07:00    M-CENTRIFUGE-01     69.2    2.14      96.3          1200       NORMAL
2024-12-10 11:00    M-CENTRIFUGE-01     82.1    5.87      34.1          31         CRITICAL
```

### `warehouse_inventory.csv` (900 baris)
10 SKU × 90 hari. Termasuk spare parts yang sengaja dibuat hampir habis menjelang maintenance.

### `production_quality.csv` (360 baris)
4 production lines × 90 hari. Termasuk quality event (bad batch) yang berkorelasi dengan kondisi mesin.

---

## 🛠️ Tech Stack (100% Gratis)

| Layer | Tool | Kenapa |
|---|---|---|
| **LLM** | Groq free tier (Llama 3.1 70B) | 6.000 token/menit gratis |
| **Agent Framework** | LangGraph | State machine, production-grade |
| **Anomaly Detection** | scikit-learn Isolation Forest | Unsupervised, no labels needed |
| **RUL Prediction** | Gradient Boosting Regressor | Robust untuk time series degradation |
| **Demand Forecast** | Prophet (optional) / ExpSmoothing | Industry standard |
| **Dashboard** | Streamlit | Deploy gratis di Streamlit Cloud |
| **Visualization** | Plotly | Interactive charts |
| **Data** | Simulated CSV + DuckDB | Zero cloud cost |

**Estimasi biaya bulanan: Rp 0,-**

---

## 🚀 Cara Menjalankan

```bash
# 1. Clone dan install
git clone https://github.com/USERNAME/wilmar-plant-intelligence.git
cd wilmar-plant-intelligence
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Setup API (gratis di console.groq.com)
cp .env.example .env
# Edit .env: isi GROQ_API_KEY

# 3. Generate data simulasi
python main.py --setup

# 4. Coba semua mode
python main.py --demo                                    # demo CLI
python main.py -q "Mesin mana yang paling kritis?"      # query langsung
python main.py --module cross                            # modul spesifik
streamlit run dashboard/app.py                           # dashboard
```

---

## 💼 Business Value

Berdasarkan benchmark industri CPO Indonesia:

| Masalah | Solusi AI | Estimasi Value |
|---|---|---|
| Unplanned downtime 1 mesin = 8 jam | Early warning 48-168 jam sebelumnya | Hemat Rp 300-500 juta/incident |
| Manual quality check = 2 jam/shift | Automated anomaly detection real-time | 4 jam/hari analyst time saved |
| Excess/shortage spare part | EOQ-based reorder recommendation | Inventory cost -15-20% |
| Root cause analysis = 2-3 hari | AI correlation < 1 menit | Faster corrective action |

---

## 📁 Struktur Folder

```
wilmar-plant-intelligence/
├── 🤖 agents/
│   ├── warehouse/          # Demand forecast, reorder, expiry
│   ├── maintenance/        # Sensor monitor, RUL predictor, scheduler
│   └── quality/            # Anomaly detect, root cause, yield forecast
├── 🧠 core/
│   ├── orchestrator.py     # LangGraph state machine (CEO agent)
│   ├── cross_domain.py     # Cross-module insight engine ⭐
│   ├── alert_engine.py     # Centralised alert routing
│   └── llm_client.py       # Groq/Anthropic wrapper
├── 📊 dashboard/           # Streamlit app (5 tabs)
├── 📦 data/simulated/      # Generated datasets
├── 🧪 tests/               # Unit tests per modul
├── generate_data.py        # Dataset simulator
└── main.py                 # CLI entry point
```

---

## 📬 Kontak

- 📧 Email: [xian.adikusuma@gmail.com]

---

<sub>Portfolio project — tidak berafiliasi dengan Wilmar International Ltd. Data adalah simulasi.</sub>
