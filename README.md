# LXCP: approximate reversible adder for QRS detection

Code, results and figures for the manuscript *"LXCP: A Toffoli-Efficient Approximate Reversible Adder
with Carry Prediction for Error-Tolerant QRS Detection"* ([authors / venue to be added]).

LXCP replaces the `k` least-significant bit positions of an adder with single CNOT gates, predicts the carry
into the exact upper part with one Toffoli gate, and completes the sum with either a Cuccaro-type adder (**LXCP**)
or a one-Toffoli-per-bit carry-ancilla adder (**LXCP-G**). The circuits are classical reversible (X, CNOT, Toffoli)
and are simulated exactly; nothing here was run on quantum hardware.

## Repository layout

| Path | Content |
|---|---|
| `src/lxcp_adder.py` | Circuit builders (LXCP, LXCP-G, lower-XOR), exact simulator, exhaustive 8-bit verification, gate counts, OpenQASM export |
| `src/qrs_approx_adders_v2.py` | QRS detection on MIT-BIH with exact addition, five approximate full-adder cells and LXCP / LXCP-G |
| `src/noise_mc.py` | Pauli-error Monte Carlo noise study of 8-bit adders |
| `src/design_search.py` | Exhaustive search over CNOT-only low-part variants (design exploration) |
| `src/make_figures.py` | Regenerates Fig. 1 (block diagram) and Fig. 2 (circuits) |
| `results/` | `qrs_summary.csv`, `qrs_per_record.csv`, `lxcp_8bit_comparison.csv`, `noise_results.json` |
| `figures/` | Figures used in the manuscript |

## Install

```bash
pip install -r requirements.txt
```

Python 3.9 or later. Qiskit is optional: if it is not installed, a small built-in circuit class is used.

## Reproduce

Run everything from the repository root.

```bash
# 1. Adder circuits: exhaustive verification (65,536 input pairs), gate counts, error metrics, QASM example
python src/lxcp_adder.py            # prints "all circuits verified", writes results/lxcp_8bit_comparison.csv

# 2. Noise study (about a minute)
python src/noise_mc.py              # writes results/noise_results.json

# 3. QRS experiment (downloads MIT-BIH records 100, 101, 103, 105, 119, 200 from PhysioNet)
python src/qrs_approx_adders_v2.py  # writes qrs_results/qrs_summary.csv, qrs_per_record.csv and figures

# 4. Figures 1 and 2
python src/make_figures.py
```

Checks printed by step 3 before the experiment starts: `adder self-test passed` and
`exact adder-tree Toffolis (Cuccaro): 946 (expected 946)`.

To run the full database, set `RECORDS = wfdb.get_record_list('mitdb')` and `SECONDS = None` at the top of
`src/qrs_approx_adders_v2.py` (several hours). `OFFSET_REMOVE`, `KS` and `APPROX_CELLS` are also set there.

## Data

The MIT-BIH Arrhythmia Database is public on PhysioNet and is **not** redistributed here. The script downloads
the required records into `mitdb/` at run time (internet access needed).

## Notes on interpretation

* Gate costs follow one transparent mapping (Toffoli = 7 T gates; Clifford+T total counts a Toffoli as 15 gates).
* Toffoli savings are reported against two exact baselines (Cuccaro-type and one-Toffoli-per-bit); see the manuscript.
* The noise study is a simple Pauli-error model, not a device-calibrated one.

## Citation

If you use this code, please cite the manuscript (details to be added) and this repository
(`https://github.com/csk-hub/lxcp-approximate-quantum-adder`).

## License

MIT, see `LICENSE` (replace the placeholder name before publishing).
