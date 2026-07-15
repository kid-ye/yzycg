# EdgeECGAuth — ECG Biometric Authentication

Reimplementation and extension of *"ECG Biometric Authentication Using Self-Supervised
Learning for IoT Edge Sensors"* (Wang et al., 2024), with a live capture path using a
**MAX30003** analog front-end and a **Raspberry Pi Pico 2**.

## Repository layout

```
mini_project_VI/
├── src/                    # Core Python package
│   ├── preprocessing.py    #   filtering + R2R / P2T / NPD segmentation
│   ├── model.py            #   1D-CNN encoder (ECGEncoder / TripletECGModel)
│   ├── data_loader.py      #   unified PTBDB / MITDB / ECGIDDB loaders (subject-level)
│   ├── ptbdb_benchmark.py  #   training + evaluation (Siamese & Triplet, EER/FAR/FRR)
│   ├── quantization.py     #   fixed-point quantization (Section V-A)
│   ├── inference.py        #   ECGAuthenticator: enroll / authenticate
│   └── live_auth.py        #   live serial authentication loop
├── firmware/
│   └── pico_max30003.py    # MicroPython firmware: MAX30003 -> UART stream (runs on Pico 2)
├── scripts/
│   ├── download_data.py    # fetch PTBDB records
│   ├── run_all.py          # train + evaluate all 6 seg×framework combos (Table I)
│   └── plots/              # figure-generation scripts
├── tests/                  # test_model.py, test_max30003.py
├── models/                 # trained weights (ecg_model.pth, enrolled_profiles.pt)
├── results/                # generated figures + score arrays
├── docs/
│   ├── paper/              # source paper (PDF/DOCX)
│   └── diagrams/           # architecture / workflow flowcharts (HTML)
└── data/                   # datasets (gitignored) — data/ptbdb, data/users
```

## Running

All scripts are meant to be run **from the repository root**:

```bash
python scripts/download_data.py          # get PTBDB
python scripts/run_all.py --epochs 20    # train + benchmark all combos
python tests/test_model.py               # sanity-check the encoder
python -m src.live_auth                   # live authentication (needs Pico on serial)
```

## Notes / known work items

- `src/ptbdb_benchmark.py` is the **correct** training path (`1 - pearson` distance,
  subject-level split, threshold tuning). The older buggy trainer was removed.
- The live path samples at ~125 Hz but the model expects 200 Hz — resampling still TODO.
- The encoder's large final FC layer (~36 MB model) must be slimmed (global pooling)
  before it can fit on a Pico 2 / ESP32.

See `docs/paper/` for the reference paper.
