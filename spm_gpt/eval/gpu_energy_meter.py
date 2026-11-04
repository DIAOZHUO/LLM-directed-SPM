"""
GPU energy measurement based on the NVML cumulative energy counter (nvmlDeviceGetTotalEnergyConsumption, mJ).

The counter is global to the GPU, so it measures both in-process inference (unsloth / transformers)
and inference in another process (LM Studio). Nothing else should run on the GPU during the measurement.

    meter = GPUEnergyMeter()
    meter.measure_idle()          # idle power, subtracted to obtain the net inference energy
    meter.reset(); meter.start()
    for example in dataset:
        with meter.measure() as rec:
            ...                       # inference of one sample
            rec["prompt_tokens"] = ...
    meter.stop()
    summary = meter.summary()

The counter is updated at the power-sampling rate of the driver, so per-sample values of very short
samples (< 1 s) are noisy; the run-level totals (total_energy_j / num_samples) are more reliable.
"""
import time
from contextlib import contextmanager

import numpy as np
import pynvml


class GPUEnergyMeter:
    def __init__(self, device_index=0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.device_name = pynvml.nvmlDeviceGetName(self.handle)
        self.idle_power_w = None
        self.reset()

    def read(self):
        """(time in s, cumulative energy in J)"""
        return time.perf_counter(), pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0

    def measure_idle(self, seconds=30):
        t0, e0 = self.read()
        time.sleep(seconds)
        t1, e1 = self.read()
        self.idle_power_w = (e1 - e0) / (t1 - t0)
        print(f"idle power: {self.idle_power_w:.1f} W")
        return self.idle_power_w

    def reset(self):
        self.records = []
        self._start = None
        self._stop = None

    def start(self):
        self._start = self.read()

    def stop(self):
        self._stop = self.read()

    @contextmanager
    def measure(self):
        rec = {}
        t0, e0 = self.read()
        try:
            yield rec
        finally:
            t1, e1 = self.read()
            rec["latency_s"] = t1 - t0
            rec["energy_j"] = e1 - e0
            self.records.append(rec)

    def gpu_memory_used_mb(self):
        return int(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1024 ** 2)

    def summary(self):
        idle = self.idle_power_w or 0.0
        latency = [r["latency_s"] for r in self.records]
        energy = [r["energy_j"] for r in self.records]
        n = len(self.records)

        result = {
            "device": self.device_name,
            "idle_power_w": self.idle_power_w,
            "num_measured_samples": n,
            "latency_list": latency,
            "latency_mean": float(np.mean(latency)) if n else None,
            "energy_list_j": energy,
            "net_energy_list_j": [e - idle * t for e, t in zip(energy, latency)],
        }
        for key in ("prompt_tokens", "completion_tokens"):
            if n and all(key in r for r in self.records):
                result[f"{key}_list"] = [r[key] for r in self.records]
                result[f"{key}_mean"] = float(np.mean(result[f"{key}_list"]))

        # run-level totals (include the overhead between samples)
        if self._start is not None and self._stop is not None:
            total_time = self._stop[0] - self._start[0]
            total_energy = self._stop[1] - self._start[1]
            net_total_energy = total_energy - idle * total_time
            result.update({
                "total_time_s": total_time,
                "total_energy_j": total_energy,
                "net_total_energy_j": net_total_energy,
                "total_energy_mwh": total_energy / 3.6,
                "net_total_energy_mwh": net_total_energy / 3.6,
                "mean_power_w": total_energy / total_time,
                "energy_per_sample_j": total_energy / n if n else None,
                "net_energy_per_sample_j": net_total_energy / n if n else None,
            })
        return result
