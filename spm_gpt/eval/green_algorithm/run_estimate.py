"""
Read gpt4.json and Phi-4(distilled).json, count output tokens,
then run Green Algorithms energy estimation.

Token counting: len(text) / 4  (standard approximation: 1 token ≈ 4 chars)

Local (Phi-4):  energy = PUE x TDP x usage x runtime  (GA methodology)
Cloud (o4-mini): energy estimated via published kWh/token benchmarks.
  - Cloud API tps reflects streaming rate to user, NOT server compute time.
  - Correct baseline: Samsi et al. 2023 "From Words to Watts" (LLaMA-65B on 8xA100)
    and Luccioni et al. 2023 "Power Hungry Processing" (measured per-token energy).

Ref: Lannelongue et al. 2021, Adv. Sci. doi:10.1002/advs.202100707
"""

import json
from pathlib import Path

# ── Local constants (Green Algorithms methodology) ───────────────────────────
RTX5090_TDP_W  = 575.0   # watts
RTX5090_USAGE  = 0.35    # inference utilization (well below TDP)
LOCAL_PUE      = 1.0     # no data center overhead
LOCAL_CI       = 0.45    # kgCO2/kWh, Japan grid (IEA 2024)

# ── Cloud constants (per-token energy from literature) ───────────────────────
# Samsi et al. 2023 "From Words to Watts":
#   LLaMA-65B on 8×A100: ~0.017 kWh/1k tokens (measured baseline)
#   Upper bound extrapolated to larger model class (GPT-4 scale): ~0.030 kWh/1k tokens
CLOUD_KWH_PER_1K_LOW  = 0.017   # kWh/1k tokens — LLaMA-65B on 8×A100 (Samsi et al. 2023)
CLOUD_KWH_PER_1K_HIGH = 0.030   # kWh/1k tokens — upper bound for larger model class
CLOUD_PUE  = 1.3    # typical hyperscale data center
CLOUD_CI   = 0.40   # kgCO2/kWh, US data center mix (with some renewables)


def count_tokens(predictions: list) -> int:
    """Approximate token count: 1 token ≈ 4 characters."""
    return sum(len(p) for p in predictions) // 4


def ga_energy_kwh(tdp_w, runtime_h, usage=1.0, pue=1.0) -> float:
    return pue * tdp_w * usage * runtime_h / 1000.0


def co2_g(energy_kwh, ci_kg_per_kwh) -> float:
    return energy_kwh * ci_kg_per_kwh * 1000.0


def load_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    base = Path(__file__).parent.parent
    phi = load_json(base / "eval_results/Phi-4(distilled).json")
    gpt = load_json(base / "eval_results/gpt4.json")

    # phi_tokens = count_tokens(phi["predictions"])
    # gpt_tokens = count_tokens(gpt["predictions"])
    # print(phi_tokens, gpt_tokens)
    phi_tokens = 154095
    gpt_tokens = 168293

    phi_tps    = phi["token_per_second"]
    gpt_tps    = gpt["token_per_second"]

    # ── Local: GA methodology (TDP × usage × runtime) ──────────────────────
    phi_runtime_h = (phi_tokens / phi_tps) / 3600.0
    local_e   = ga_energy_kwh(RTX5090_TDP_W, phi_runtime_h, RTX5090_USAGE, LOCAL_PUE)
    local_co2 = co2_g(local_e, LOCAL_CI)

    # ── Cloud: per-token energy from literature ─────────────────────────────
    cloud_e_low   = (gpt_tokens / 1000) * CLOUD_KWH_PER_1K_LOW  * CLOUD_PUE
    cloud_e_high  = (gpt_tokens / 1000) * CLOUD_KWH_PER_1K_HIGH * CLOUD_PUE
    cloud_co2_low  = co2_g(cloud_e_low,  CLOUD_CI)
    cloud_co2_high = co2_g(cloud_e_high, CLOUD_CI)

    ratio_e_low  = cloud_e_low  / local_e
    ratio_e_high = cloud_e_high / local_e
    ratio_c_low  = cloud_co2_low  / local_co2
    ratio_c_high = cloud_co2_high / local_co2

    sep = "=" * 62
    print(f"\n{sep}")
    print("  Green Algorithms — LLM Inference Energy Estimation")
    print("  Lannelongue et al. 2021, Adv. Sci. doi:10.1002/advs.202100707")
    print(sep)
    print(f"  Dataset: {phi_tokens:,} output tokens\n")

    print("[Local — Phi-4 (distilled) on RTX 5090]")
    print(f"  Output tokens   : {phi_tokens:,}")
    print(f"  Tokens/s        : {phi_tps:.2f}")
    print(f"  Runtime         : {phi_runtime_h*3600:.1f} s")
    print(f"  TDP × usage     : {RTX5090_TDP_W:.0f} W × {RTX5090_USAGE}  |  PUE {LOCAL_PUE}")
    print(f"  Energy          : {local_e*1000:.4f} mWh  ({local_e:.6f} kWh)")
    print(f"  CO2             : {local_co2:.3f} gCO2eq  (CI={LOCAL_CI} kgCO2/kWh, Japan)")

    print(f"\n[Cloud — o4-mini / GPT-4  (per-token energy from literature)]")
    print(f"  Output tokens   : {gpt_tokens:,}")
    print(f"  kWh/1k tokens   : {CLOUD_KWH_PER_1K_LOW}–{CLOUD_KWH_PER_1K_HIGH}  |  PUE {CLOUD_PUE}")
    print(f"  Energy          : {cloud_e_low*1000:.3f}–{cloud_e_high*1000:.3f} mWh")
    print(f"  CO2             : {cloud_co2_low:.3f}–{cloud_co2_high:.3f} gCO2eq  (CI={CLOUD_CI} kgCO2/kWh, US)")
    print( "  Ref: Samsi et al. 2023 'From Words to Watts' (LLaMA-65B on 8xA100)")
    print(f"  Note: cloud tps ({gpt_tps:.1f}) reflects API streaming, not server compute")

    print(f"\n[Ratio: Cloud / Local]")
    print(f"  Energy          : {ratio_e_low:.1f}x – {ratio_e_high:.1f}x")
    print(f"  CO2             : {ratio_c_low:.1f}x – {ratio_c_high:.1f}x")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
