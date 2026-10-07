# -*- coding: utf-8 -*-
"""Small helper for printing verification scores."""


def print_score_summary(ec_results, unet_results, title='Verification scores'):
    """Print HSS / TS / POD / FAR of the raw ECMWF forecasts and the U-Net.

    Parameters
    ----------
    ec_results, unet_results : dict
        {lead week: {'hss': ..., 'ts': ..., 'pod': ..., 'far': ...}}
    """
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)
    print(f"{'lead':<6}{'method':<9}{'HSS':>9}{'TS':>9}{'POD':>9}{'FAR':>9}")
    for lead in sorted(set(ec_results) | set(unet_results)):
        for name, results in (('ECMWF', ec_results), ('U-Net', unet_results)):
            if lead not in results:
                continue
            m = results[lead]
            print(f"{lead:<6}{name:<9}{m['hss']:>9.3f}{m['ts']:>9.3f}"
                  f"{m['pod']:>9.3f}{m['far']:>9.3f}")
    print("=" * 62)
