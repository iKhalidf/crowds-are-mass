# GCS results

Clips: 1448. Kappa: 8. Lower is better.

Strata use future-count range, with quartiles 8.0325, 10.7683, 15.2657 people.

ADJS follows CrowdMAC's reference mixture computation, not standard normalized JS.
Unmatched mass depends on both count and placement.

| Stratum | n | Model | countMAE | D_kappa | Unmatched (people) | Transport | ADJS |
|---|---:|---|---:|---:|---:|---:|---:|
| flat | 363 | Stream-CDF | 3.59 | 203.3 | 12.01 | 107.2 | 0.1446 |
| flat | 363 | persistence | 3.27 | 209.6 | 12.45 | 110.0 | 0.0818 |
| flat | 363 | CrowdMAC | 5.49 | 293.0 | 19.42 | 137.7 | 0.1087 |
| flat | 363 | Holt + last | 8.72 | 222.9 | 15.13 | 101.9 | 0.0836 |
| flat | 363 | Holt + uniform | 8.72 | 615.4 | 63.27 | 109.3 | 0.2919 |
| moderate | 361 | Stream-CDF | 4.41 | 228.4 | 13.51 | 120.3 | 0.1274 |
| moderate | 361 | persistence | 4.22 | 235.7 | 13.98 | 123.9 | 0.0719 |
| moderate | 361 | CrowdMAC | 7.17 | 333.5 | 22.19 | 155.9 | 0.0979 |
| moderate | 361 | Holt + last | 9.86 | 252.4 | 16.92 | 117.0 | 0.0740 |
| moderate | 361 | Holt + uniform | 9.86 | 729.4 | 75.65 | 124.2 | 0.2879 |
| dynamic | 363 | Stream-CDF | 6.00 | 258.0 | 15.73 | 132.2 | 0.1070 |
| dynamic | 363 | persistence | 5.82 | 264.4 | 16.07 | 135.9 | 0.0594 |
| dynamic | 363 | CrowdMAC | 8.31 | 374.7 | 24.52 | 178.6 | 0.0872 |
| dynamic | 363 | Holt + last | 10.26 | 280.5 | 18.71 | 130.8 | 0.0639 |
| dynamic | 363 | Holt + uniform | 10.26 | 861.4 | 89.89 | 142.3 | 0.2848 |
| surge | 361 | Stream-CDF | 10.63 | 335.6 | 22.05 | 159.3 | 0.0808 |
| surge | 361 | persistence | 10.39 | 341.8 | 22.37 | 162.9 | 0.0452 |
| surge | 361 | CrowdMAC | 28.25 | 570.7 | 43.68 | 221.3 | 0.0790 |
| surge | 361 | Holt + last | 14.19 | 358.1 | 25.01 | 158.0 | 0.0454 |
| surge | 361 | Holt + uniform | 14.19 | 1268.4 | 134.35 | 193.6 | 0.2730 |
| all | 1448 | Stream-CDF | 6.16 | 256.3 | 15.82 | 129.7 | 0.1150 |
| all | 1448 | persistence | 5.92 | 262.8 | 16.21 | 133.1 | 0.0646 |
| all | 1448 | CrowdMAC | 12.29 | 392.8 | 27.44 | 173.3 | 0.0932 |
| all | 1448 | Holt + last | 10.75 | 278.4 | 18.94 | 126.9 | 0.0667 |
| all | 1448 | Holt + uniform | 10.75 | 868.3 | 90.75 | 142.3 | 0.2844 |

Reconstructed from stored per-clip arrays; no training or full-map transport was rerun.
