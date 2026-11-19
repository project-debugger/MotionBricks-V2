# Animation clean-up ablation

`benchmarks/ablation_postprocess.py --model soma_int8_pose` on all 187 held-out eval windows (duration pinned to the ground-truth gap, so every frame is compared with mocap).

| variant | what | MPJPE cm | foot slide cm/s (height contacts) | foot slide cm/s (model contacts) | jerk m/s^3 | join pop start / end | end-pose error cm |
|---|---|---|---|---|---|---|---|
| A_raw | raw network output | 8.21 | 10.81 | 8.50 | 116.3 | 13.48 / 14.02 | 3.51 |
| B_upstream | + upstream endpoint correction (NVIDIA engine behaviour) | 7.08 | 12.49 | 10.63 | 122.7 | 3.68 / 3.72 | 0.00 |
| C_velocity | + velocity-matched joins (somart) | 6.70 | 12.92 | 11.70 | 126.8 | 1.91 / 1.98 | 0.02 |
| D_footlock | upstream + foot locking (somart) | 7.07 | 7.95 | 5.16 | 125.9 | 3.61 / 3.96 | 0.10 |
| E_both | upstream + velocity matching + foot locking (**somart default**) | 6.69 | 7.81 | 5.72 | 130.8 | 1.89 / 2.23 | 0.10 |
| F_contact | contact-aware root ramp instead of linear (somart, experimental) | 7.09 | 12.55 | 10.79 | 124.7 | 3.67 / 3.73 | 0.00 |
| G_full | contact-aware + velocity + foot locking | 6.69 | 7.81 | 5.79 | 132.7 | 1.89 / 2.22 | 0.10 |
| GT | held-out mocap (reference level) | 0.00 | 5.16 | - | 158.0 | 1.15 / 1.29 | 0.00 |

## Result

- Velocity matching + foot locking vs upstream correction: MPJPE 7.08 -> 6.69 cm (-5.5 %), foot sliding 12.49 -> 7.81 cm/s (-37.5 %), join pops 3.68/3.72 -> 1.89/2.23 (-49 % / -40 %).
- Upstream endpoint correction removes the raw endpoint error and the large raw join pops, but *increases* foot sliding (it spreads leg-joint corrections over the segment); foot locking removes that again.
- Contact-aware root ramp (F): foot sliding 12.55 vs 12.49 cm/s -> no measurable benefit; kept as an option, not enabled (negative result).
- Remaining gap to mocap: foot sliding 1.5x and join pops 1.6x / 1.7x the ground-truth level.
