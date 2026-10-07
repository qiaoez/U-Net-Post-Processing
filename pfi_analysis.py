# -*- coding: utf-8 -*-
"""Permutation feature importance (PFI) analysis.

For each lead week the trained U-Net model is loaded and the HSS-based PFI of
every input channel is computed over the test period (2017-2020).  The result is
written to a pickle file so that it can be inspected afterwards.
"""
import os
import pickle
import sys

import numpy as np
import torch

# allow running the script from outside this folder
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import *                                        # noqa: F401,F403
from data_processing import WeeklyEC_S2S_Dataset
from interpretability import calculate_pfi_for_model
from models import UNet

LEADS = [1, 2, 3, 4]
IN_CHANNELS = 20


def calculate_pfi_for_all_models(dataset, test_years, device):
    """Return {lead week: [importance of each input channel]}."""
    print("\nPermutation feature importance (PFI) analysis")
    pfi_results = {}

    for lead_time in LEADS:
        print(f"\nLead week {lead_time} ...")
        weight_path = os.path.join(
            MODEL_SAVE_DIR, f'unet_lead_{lead_time}_best.pth')
        if not os.path.exists(weight_path):
            print(f"  model not found: {weight_path}, skipped")
            continue

        model = UNet(in_channels=IN_CHANNELS, out_channels=1).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))

        pfi_results[lead_time] = calculate_pfi_for_model(
            model, dataset, test_years, metric='hss', device=device)

    return pfi_results


def main_interpretability_analysis(dataset=None, test_years=TEST_YEARS,
                                   save_path=None):
    """Run the PFI analysis for the four lead-week models."""
    print("\n" + "=" * 60)
    print("Interpretability analysis - PFI")
    print("=" * 60)

    if dataset is None:
        print("\nLoading data set ...")
        dataset = WeeklyEC_S2S_Dataset(start_year=2001, end_year=2020)

    device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA
                          else 'cpu')
    print(f"Device: {device}")

    pfi_results = calculate_pfi_for_all_models(dataset, test_years, device)

    if pfi_results:
        if save_path is None:
            save_path = os.path.join(MODEL_SAVE_DIR, 'pfi_results.pkl')
        with open(save_path, 'wb') as handle:
            pickle.dump(pfi_results, handle)
        print(f"\nPFI results written to {save_path}")
        for lead_time, importance in pfi_results.items():
            print(f"  lead week {lead_time}: most important channel = "
                  f"{int(np.argmax(importance))} "
                  f"(importance {np.nanmax(importance):.4f})")
    else:
        print("Warning: no valid PFI result")

    print("=" * 60)
    return pfi_results


if __name__ == "__main__":
    main_interpretability_analysis()
