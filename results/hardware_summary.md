# IBM Quantum hardware run

* Submitted: 2026-11-16 01:41:36; analysed: 2026-11-16 01:44:53
* QPU: ibm_marrakesh; shots: 1024; dynamical decoupling: False; twirling: False
* QPU time used: 189 s
* Jobs are also on the IBM Quantum Workloads page: https://quantum.cloud.ibm.com/workloads

## Jobs

| module | job ID | QPU | status | QPU seconds | two-qubit gates |
|---|---|---|---|---|---|
| m1_p1 | `dbtcvkfasvbs73b9je9g` | ibm_marrakesh | done | 4.0 | 202 |
| m1_p3 | `dbtcvkhvvpic738mqqtg` | ibm_marrakesh | done | 4.0 | 675 |
| m2_p1 | `dbtcvkhvvpic738mqqu0` | ibm_marrakesh | done | 13.0 | 166 |
| m2_p3 | `dbtcvkvq0h0s73bkk860` | ibm_marrakesh | done | 14.0 | 508 |
| m3_kernel | `dbtcvl3eq3sc73evllf0` | ibm_marrakesh | done | 70.0 | 22 |
| m4_kernel | `dbtcvl7asvbs73b9jeb0` | ibm_marrakesh | done | 71.0 | 34 |
| m5_policy | `dbtcvl9vvpic738mqqv0` | ibm_marrakesh | done | 13.0 | 56 |

## Ideal vs noisy vs hardware

| | exact | ideal Aer | noisy FakeFez | IBM ibm_marrakesh | uniform |
|---|---|---|---|---|---|
| M1 P(opt), p=1 | 0.0053 | 0.0049 | 0.0042 | 0.0027 | 0.0010 |
| M1 P(opt), p=3 | 0.0155 | 0.0163 | 0.0043 | 0.0010 | 0.0010 |
| M2 P(opt), p=1 | 0.0045 | 0.0040 | 0.0029 | 0.0017 | 0.0010 |
| M2 P(opt), p=3 | 0.0132 | 0.0138 | 0.0045 | 0.0032 | 0.0010 |
| M3 kernel MAE | 0 | 0.007 | 0.017 | 0.024 | – |
| M4 kernel MAE | 0 | 0.008 | 0.062 | 0.105 | – |
| M5 gain error | 0 | 1.1 % | 5.6 % | 15.1 % | – |

## Per module

* **m1_p1**: P(optimum) 0.0027 (exact 0.0053, uniform 0.0010); approximation ratio 0.795 (exact 0.889); best sample = optimum in 6/7 instances
* **m1_p3**: P(optimum) 0.0010 (exact 0.0155, uniform 0.0010); approximation ratio 0.696 (exact 0.915); best sample = optimum in 5/7 instances
* **m2_p1**: P(optimum) 0.0017 (exact 0.0045, uniform 0.0010); approximation ratio 0.778 (exact 0.873); best sample = optimum in 33/40 instances
* **m2_p3**: P(optimum) 0.0032 (exact 0.0132, uniform 0.0010); approximation ratio 0.805 (exact 0.929); best sample = optimum in 40/40 instances
* **m3_kernel**: kernel mean abs error 0.0236, max 0.138 (mean exact entry 0.18); GP error with exact kernel 0.326, with device kernel 0.310
* **m4_kernel**: kernel mean abs error 0.1050, max 0.263 (mean exact entry 0.47)
* **m5_policy**: gain error 15.1 %, <Z> mean abs error 0.1898

## Files

* `results/hardware_ibm.json`: all metrics above, plus circuit sizes
* `results/hardware_ibm_counts.json.gz`: every measured bitstring count
* `results/ibm_jobs.json`: job IDs (retrieve any job later with `service.job(<id>)`)
* `figures/fig5_hardware.png`: ideal vs noisy vs hardware chart
