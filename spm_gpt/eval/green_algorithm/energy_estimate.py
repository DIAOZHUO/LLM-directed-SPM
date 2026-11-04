"""
LLM inference energy & CO2 estimator.

Methodology: Green Algorithms (Lannelongue et al. 2021, Advanced Science)
  Energy (Wh) = PUE x n_GPUs x TDP_W x usage_factor x runtime_h
  CO2 (gCO2eq) = Energy_kWh x carbon_intensity_gCO2_per_kWh

Online calculator: https://calculator.green-algorithms.org/ai

Usage:
    python energy_estimate.py \
        --local_tokens 500  --local_tps 35 \
        --cloud_tokens 800  --cloud_tps 60

Optional overrides:
    --gpu_tdp_w        GPU TDP in watts         (default: 575, RTX 5090)
    --gpu_usage        GPU utilization factor   (default: 0.35 for inference)
    --local_pue        Local PUE                (default: 1.0, no data center)
    --cloud_pue        Cloud data center PUE    (default: 1.3)
    --local_ci         Local carbon intensity   (default: 0.45 kgCO2/kWh, Japan)
    --cloud_ci         Cloud carbon intensity   (default: 0.40 kgCO2/kWh, US mix)
    --n_queries        Number of queries (for per-query breakdown)
"""

import argparse


def green_algorithms_energy(
    tdp_w: float,
    runtime_h: float,
    n_gpus: int = 1,
    usage_factor: float = 1.0,
    pue: float = 1.0,
) -> float:
    """
    Compute energy in kWh following the Green Algorithms methodology.
    Ref: Lannelongue et al. 2021, Advanced Science, doi:10.1002/advs.202100707
    """
    power_w = pue * n_gpus * tdp_w * usage_factor
    energy_kwh = power_w * runtime_h / 1000.0
    return energy_kwh


def co2(energy_kwh: float, carbon_intensity_kg_per_kwh: float) -> float:
    """Return CO2 in grams."""
    return energy_kwh * carbon_intensity_kg_per_kwh * 1000.0


def main():
    parser = argparse.ArgumentParser(
        description="Green-Algorithms-based LLM inference energy estimator"
    )
    # Required
    parser.add_argument("--local_tokens", type=float, required=True,
                        help="Total output tokens from local Phi-4")
    parser.add_argument("--local_tps",    type=float, required=True,
                        help="Tokens/s of local Phi-4")
    parser.add_argument("--cloud_tokens", type=float, required=True,
                        help="Total output tokens from cloud LLM (o4-mini)")
    parser.add_argument("--cloud_tps",    type=float, required=True,
                        help="Tokens/s of cloud LLM")

    # Optional hardware / location
    parser.add_argument("--gpu_tdp_w",  type=float, default=575.0,
                        help="GPU TDP in watts (default: 575, RTX 5090)")
    parser.add_argument("--gpu_usage",  type=float, default=0.35,
                        help="GPU utilization factor during inference (default: 0.35)")
    parser.add_argument("--local_pue",  type=float, default=1.0,
                        help="PUE for local deployment (default: 1.0)")
    parser.add_argument("--cloud_pue",  type=float, default=1.3,
                        help="PUE for cloud data center (default: 1.3)")
    parser.add_argument("--local_ci",   type=float, default=0.45,
                        help="Local carbon intensity kgCO2/kWh (default: 0.45, Japan)")
    parser.add_argument("--cloud_ci",   type=float, default=0.40,
                        help="Cloud carbon intensity kgCO2/kWh (default: 0.40, US mix)")
    parser.add_argument("--n_queries",  type=int, default=None,
                        help="Number of queries (for per-query breakdown)")

    args = parser.parse_args()

    # --- Local (Phi-4 on RTX 5090) ---
    local_runtime_h = (args.local_tokens / args.local_tps) / 3600.0
    local_energy    = green_algorithms_energy(
        tdp_w        = args.gpu_tdp_w,
        runtime_h    = local_runtime_h,
        usage_factor = args.gpu_usage,
        pue          = args.local_pue,
    )
    local_co2 = co2(local_energy, args.local_ci)

    # --- Cloud (o4-mini) ---
    # Cloud energy estimated from Lannelongue et al. + published LLM inference benchmarks.
    # We use the measured inference time (tokens / tps) as the billed compute time,
    # and apply a range of power estimates for large-scale GPU clusters.
    cloud_runtime_h     = (args.cloud_tokens / args.cloud_tps) / 3600.0
    CLOUD_GPU_TDP_LOW   = 400.0   # A100 SXM4 (conservative server GPU)
    CLOUD_GPU_TDP_HIGH  = 700.0   # H100 SXM5 (high-end)
    CLOUD_USAGE         = 0.60    # server GPUs run at higher utilization

    cloud_energy_low  = green_algorithms_energy(
        tdp_w=CLOUD_GPU_TDP_LOW, runtime_h=cloud_runtime_h,
        usage_factor=CLOUD_USAGE, pue=args.cloud_pue)
    cloud_energy_high = green_algorithms_energy(
        tdp_w=CLOUD_GPU_TDP_HIGH, runtime_h=cloud_runtime_h,
        usage_factor=CLOUD_USAGE, pue=args.cloud_pue)

    cloud_co2_low  = co2(cloud_energy_low,  args.cloud_ci)
    cloud_co2_high = co2(cloud_energy_high, args.cloud_ci)

    ratio_e_low  = cloud_energy_low  / local_energy
    ratio_e_high = cloud_energy_high / local_energy
    ratio_c_low  = cloud_co2_low  / local_co2
    ratio_c_high = cloud_co2_high / local_co2

    # --- Print ---
    sep = "=" * 58
    print(f"\n{sep}")
    print("  Green Algorithms — LLM Inference Energy Estimator")
    print(f"  Ref: Lannelongue et al. 2021, Adv. Sci. (doi:10.1002/advs.202100707)")
    print(sep)

    print(f"\n[Local — on RTX 5090]")
    print(f"  Output tokens    : {args.local_tokens:.0f}")
    print(f"  Tokens/s         : {args.local_tps:.1f}")
    print(f"  Inference time   : {local_runtime_h*3600:.1f} s  ({local_runtime_h*1000:.2f} mh)")
    print(f"  GPU TDP          : {args.gpu_tdp_w:.0f} W  |  usage factor: {args.gpu_usage:.2f}  |  PUE: {args.local_pue:.1f}")
    print(f"  Energy           : {local_energy*1e6:.2f} µWh  ({local_energy*1000:.4f} mWh)")
    print(f"  CO2              : {local_co2:.4f} gCO2eq  (CI = {args.local_ci} kgCO2/kWh, Japan)")

    print(f"\n[Cloud — o4-mini (estimated)]")
    print(f"  Output tokens    : {args.cloud_tokens:.0f}")
    print(f"  Tokens/s         : {args.cloud_tps:.1f}")
    print(f"  Inference time   : {cloud_runtime_h*3600:.1f} s  ({cloud_runtime_h*1000:.2f} mh)")
    print(f"  GPU TDP range    : {CLOUD_GPU_TDP_LOW:.0f}–{CLOUD_GPU_TDP_HIGH:.0f} W  |  usage: {CLOUD_USAGE:.2f}  |  PUE: {args.cloud_pue:.1f}")
    print(f"  Energy           : {cloud_energy_low*1e6:.2f}–{cloud_energy_high*1e6:.2f} µWh")
    print(f"  CO2              : {cloud_co2_low:.4f}–{cloud_co2_high:.4f} gCO2eq  (CI = {args.cloud_ci} kgCO2/kWh, US)")

    print(f"\n[Ratio: Cloud / Local]")
    print(f"  Energy           : {ratio_e_low:.1f}x – {ratio_e_high:.1f}x")
    print(f"  CO2              : {ratio_c_low:.1f}x – {ratio_c_high:.1f}x")

    if args.n_queries:
        print(f"\n[Scaled to {args.n_queries} queries]")
        print(f"  Local  CO2 total : {local_co2  * args.n_queries:.3f} gCO2eq")
        print(f"  Cloud  CO2 total : {cloud_co2_low * args.n_queries:.3f}–{cloud_co2_high * args.n_queries:.3f} gCO2eq")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
