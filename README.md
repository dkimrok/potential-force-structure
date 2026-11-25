# Replication materials: posture ledger and code

**The Potential Force Structure under Demographic Decline: How Full-Scale-War Risk Acceptance Governs the Value of Personnel Flexibility in the ROK Army**

This repository has two parts:

1. **`posture_ledger.xlsx`.** This is the posture ledger cited in Appendix A: the year-by-year posture coding for 1958–2023. Table A1 and the transition matrix in Table 1 are built from it.
2. **`replication/`.** This holds the model and the scripts that produce the manuscript's tables, figures, and reported numbers, together with their outputs.

## 1. File structure

```
README.md                       this file
posture_ledger.xlsx             the posture ledger (sheets "Ledger" and "Notes")
replication/
  requirements.txt              Python packages and the versions used
  model/
    potential_structure_lp.py   the linear program (Section 3) and the calibration (Section 4)
  data/
    posture_ledger.csv          the posture ledger as CSV (same content as the Ledger sheet)
    posture_coding.json         the same coding as numbers (1 peacetime, 2 tension, 3 limited conflict)
    posture_coding_alt.json     alternative coding with 1980 and 1995 as peacetime (Section 5.4)
    build_ledger.py             builds the ledger and the coding from the source databases (Section 6)
    markov_check.py             recounts the transitions (Table A1) and compares them with the model's matrix
  analysis/
    tables.py                   Tables 1 and 4–9 and most numbers in Sections 5.1–5.4   -> results.json
    supplement.py               further numbers in Sections 4.5 and 5.2–5.4             -> supplement.json
    diagnostics.py              further checks in Sections 5.1, 5.3, and 5.4            -> diagnostics.json
    coding_sensitivity.py       stricter coding and war-occupancy tests (Section 5.4)   -> coding_sensitivity.json
    moved_peaks.py              requirement profiles with moved peaks (Section 5.4)     -> moved_peaks.json
    perturb.py                  one-at-a-time perturbations at N0 along gamma           -> perturb.json
    theta_thresholds.py         weight thresholds in Findings 2 and 4                   -> theta_thresholds.json
    reserve_test.py             standing-reserve model; "base" mode: baseline requirement -> reserve_base.json
    reserve_summary.py          standing-reserve test, full-scale-war requirement held at
                                its benchmark level (Table 10)                          -> reserve_fixed.json
    figures.py                  Figures 1 and 2 from results.json                       -> figure1.png, figure2.png
```

Each run's printed output is saved in a `.log` file named after the script or its output. For example, `reserve_test.py base` writes `reserve_base.log`. The files `results_alt.json` and `diagnostics_alt.json` repeat the calculation with the alternative coding.

## 2. The posture ledger

`posture_ledger.xlsx` has one row per year: 66 rows covering 1958–2023.

| Column | Meaning |
|---|---|
| year | calendar year |
| posture | P = Peacetime, T = Tension, L = Limited conflict |
| criterion | the coding rule the year meets (codes below) |
| decisive_events | for L years, the event(s) that meet the rule; for P and T years, the counts used |
| csis_events | provocation events in the CSIS Beyond Parallel record |
| nuclear_tests | North Korean nuclear tests |
| ballistic_launches | ballistic missiles and space launch vehicles launched, each counted once (CNS/NTI missile-test database) |
| sources | sources for the year (abbreviations explained in the Notes sheet) |

| Criterion | Rule |
|---|---|
| L-a | a North Korean armed action that killed ROK or U.S. (United Nations Command) military or police personnel |
| L-b | a fatal engagement between the two sides' regular naval or ground combat units |
| L-c | a North Korean state-directed attack (bombing, raid, or assassination attempt) that killed five or more people or targeted the ROK head of state |
| T-nuclear | not L; at least one nuclear test |
| T-missiles | not L; three or more ballistic missiles or space launch vehicles launched |
| T-events | not L; five or more provocation events in the CSIS Beyond Parallel record |
| P | none of the above |

**How the coding works**
- Each year is coded to the highest posture it reached. This gives 24 peacetime, 15 tension, and 27 limited-conflict years. No post-armistice year reaches full-scale war.
- Counting each pair of consecutive years gives the 65 transitions of Table A1.
- The row shares of Table A1 give the peacetime and tension rows of Table 1. The limited-conflict row adds the assumed escalation probability (Section 4.3).

**How to check it**
- To check a year, filter the Ledger sheet by year or by posture.
- To recount the transitions, run `data/markov_check.py` (see Section 3 below).

## 3. How to run the code

**Requirements.**
- Python 3 with numpy, pandas, openpyxl, PuLP (with its bundled CBC solver), and matplotlib.
- The outputs here were produced with Python 3.13.18, numpy 2.5.3, pandas 3.0.5, openpyxl 3.1.5, PuLP 2.8.0, and matplotlib 3.11.2 (`replication/requirements.txt`).
- Each model is a linear program solved exactly, and a full run takes about two minutes.

```
pip install -r replication/requirements.txt
```

**Steps.** From `replication/analysis/`:

```
python tables.py
python supplement.py
python diagnostics.py
python coding_sensitivity.py
python moved_peaks.py
python perturb.py
python theta_thresholds.py
python reserve_test.py base
python reserve_summary.py
python figures.py
```

- Run `tables.py` first, because `supplement.py` and `figures.py` read `results.json`.
- `python tables.py alt` and `python diagnostics.py alt` repeat the calculation with the alternative coding.
- From `replication/data/`, `python markov_check.py` prints the transition counts of Table A1 and the matrix used in the model.

## 4. Where each result comes from

| Manuscript item | Script | Output (key) |
|---|---|---|
| Table 1; war occupancy of 4.2 percent | tables.py | results.json (`T1`) |
| Table A1 | data/markov_check.py | printed counts |
| Benchmark minimum of θ (about 1.54); the last requirement unmet just below it (about 2,900) | tables.py, supplement.py | results.json (`theta_min`), supplement.json (`gap_below_theta_min`) |
| Table 4, Figure 1; γ = 0 end point (157.5 and 156.8) | tables.py, diagnostics.py | results.json (`T4`, `fig1`), diagnostics.json (`gamma0`) |
| Table 5 | tables.py | results.json (`T5`) |
| Table 6, Figure 2 | tables.py, figures.py | results.json (`T6`, `fig2`) |
| Risk-acceptance threshold; feasibility limit (245,000) | tables.py | results.json (`thresholds`, `feasibility_limit`) |
| Finding 2: θ of about 3.8 and about 5.3 | theta_thresholds.py | theta_thresholds.json |
| Decomposition of the multifunctional share (Section 5.3) | tables.py, supplement.py | results.json (`decomp`), supplement.json (`forced2_N3_theta8`) |
| Tables 7 and 8 (factorial design at N₃) | tables.py | results.json (`factorial`) |
| Finding 4: raising only the full-scale-war weight (no multifunctional share at N₃ from about 6.7) | tables.py, theta_thresholds.py | results.json (`war_only`), theta_thresholds.json |
| Finding 6: zero share from θ of about 7.97; interaction | diagnostics.py, tables.py | diagnostics.json (`N3_share_zero_theta`), results.json (`A1`) |
| Table 9 | tables.py | results.json (`T9`) |
| Section 5.4, first test: monotonic and moved-peak profiles | tables.py, diagnostics.py, moved_peaks.py | results.json (`monotone_max_k5`), diagnostics.json (`sorted_monotone_h14_N0`), moved_peaks.json |
| Section 5.4, second test: full-scale-war requirement held at its benchmark level | tables.py, supplement.py | results.json (`fixedwar`), supplement.json (`fixedwar_path`) |
| Section 5.4, third test: coding and transition weights | coding_sensitivity.py, tables.py, diagnostics.py | coding_sensitivity.json, results.json (`W_sensitivity`, `pwar_range`), results_alt.json, diagnostics.json (`table6_1986P`) |
| Section 5.4, fourth test: standing reserve, Table 10 | reserve_summary.py | reserve_fixed.json |
| Standing reserve under the baseline requirement (Section 5.4, last paragraph) | reserve_test.py base | reserve_base.json |

Tables 2 and 3, and the cost and friction parameters of Section 4.5, are inputs set in `model/potential_structure_lp.py`.

## 5. Notes on the code

- **Korean labels.** Personnel types and functions carry Korean labels inside the model:
  - k1 combat power, k2 command-and-intelligence, k3 combat support, k4 service support, k5 technical-specialist
  - j1 intelligence, j2 command and control, j3 maneuver, j4 fires, j5 protection, j6 sustainment
- **The weight θ.** `RHO_SCALE` in the model is only the default of `solve()`. Every analysis script passes θ explicitly as `rho_scale`.
- **Transition matrices.**
  - `F` is the matrix used in the manuscript.
  - `F_ALT` uses the alternative coding.
  - `F_ORIG` is the matrix from the original submission's coding, kept for comparison.
- **Force sizes.** Population and force-size figures enter as the manpower path N0–N4 (Section 4.2 and note 18).

## 6. Source data

`build_ledger.py` reads two third-party databases, which are not redistributed here. To rebuild the ledger, place them in `replication/data/source/` under these names:

| File name | Database |
|---|---|
| `csis_beyond_parallel_provocations.xlsx` | CSIS Beyond Parallel database of North Korean provocations |
| `cns_nti_missile_tests.xlsx` | CNS North Korea Missile Test Database (Nuclear Threat Initiative), sheet "Missile Tests" |

`build_ledger.py` and the ledger both record the decisive events for each limited-conflict year, with their sources. Note 19 of the manuscript lists the sources used to check individual years.
