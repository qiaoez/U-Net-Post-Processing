# U-Net post-processing of ECMWF S2S reforecasts for summer extreme precipitation over the Tibetan Plateau

This repository contains the model code used in

> *Enhancing Subseasonal-to-Seasonal Prediction of Summer Extreme Precipitation
> over the Tibetan Plateau via U-Net Post-Processing*

The framework corrects weekly-mean precipitation forecasts from the ECMWF S2S
reforecasts with a standard 2-D U-Net. The training objective combines a
normalized mean squared error with the Threat Score (TS) so that systematic
biases and extreme events are optimized at the same time.

Only the model code is provided here. Plotting routines and analyses that are
not part of the paper are not included.

## Files

| File | Description |
| --- | --- |
| `config.py` | data paths, time periods and hyperparameters |
| `data_processing.py` | `WeeklyEC_S2S_Dataset`: reads the ECMWF reforecasts, the CN05.1 observations and the DEM, builds weekly means and the grid-specific 90th-percentile threshold |
| `models.py` | U-Net architecture |
| `loss.py` | composite loss `MSE + (1 - TS)`, with the balancing weight `b` |
| `train.py` | training loop (Adam, ReduceLROnPlateau, early stopping) |
| `evaluate.py` | verification: HSS, TS, POD, FAR and the raw ECMWF baseline |
| `metrics.py` | spatial maps of TCC, RMSE, variance and per-sample ACC |
| `interpretability.py` | permutation feature importance (PFI) |
| `pfi_analysis.py` | runs the PFI analysis for the four lead weeks |
| `b_sweep.py` | sensitivity experiment for the loss weight `b` (Fig. S1) |

## Data

All network training uses the ECMWF S2S reforecasts (ensemble mean of the 11
members, 1.5° × 1.5°) and the CN05.1 gridded daily precipitation observations
regridded to the same grid. Predictors are total precipitation and six
multi-level variables (u, v, w, t, gh, q at 200/300/500 hPa) plus the DEM.

Paths are set in `config.py`:

```python
DATA_PATHS = {...}          # reforecast / observation / DEM / shapefile files
MODEL_SAVE_DIR = '...'      # where unet_lead_{lead}_best.pth is written
CACHE_DIR = '...'           # cached pre-processed samples
```

## Usage

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate the four lead-week models, and run the `b` sensitivity
experiment:

```bash
python b_sweep.py
```

Run the permutation feature importance analysis for the four trained models:

```bash
python pfi_analysis.py
```

The verification scores can be printed with the helper in `utils.py`:

```python
from utils import print_score_summary
print_score_summary(ec_results, unet_results)
```

## Notes

* Training and validation are carried out independently for lead weeks 1–4.
* The loss weight is lead-time dependent: `b = 0.5` for weeks 1–2 and
  `b = 1.3` for weeks 3–4 (`a` in the sigmoid approximation is fixed to 2).
* The 90th-percentile threshold used to define extreme events is derived from
  the observations over 2001–2015, so that the test period (2017–2020) is not
  used for its estimation.
