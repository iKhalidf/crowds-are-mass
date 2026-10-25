# INDOOR results

Clips: 361. Kappa: 8. Lower is better.

Strata use future-count range, with quartiles 13.1245, 16.7668, 20.1031 people.

ADJS follows CrowdMAC's reference mixture computation, not standard normalized JS.
Unmatched mass depends on both count and placement.

| Stratum | n | Model | countMAE | D_kappa | Unmatched (people) | Transport | ADJS |
|---|---:|---|---:|---:|---:|---:|---:|
| flat | 92 | Stream-CDF | 5.51 | 79.4 | 7.54 | 19.1 | 0.0904 |
| flat | 92 | persistence | 4.70 | 75.5 | 7.17 | 18.1 | 0.0460 |
| flat | 92 | CrowdMAC | 4.80 | 301.3 | 32.01 | 45.3 | 0.4960 |
| flat | 92 | Holt + last | 11.65 | 117.9 | 13.08 | 13.2 | 0.9701 |
| flat | 92 | Holt + uniform | 11.65 | 305.5 | 33.29 | 39.2 | 1.3842 |
| moderate | 93 | Stream-CDF | 5.35 | 87.6 | 7.86 | 24.7 | 0.0921 |
| moderate | 93 | persistence | 5.75 | 88.3 | 8.33 | 21.7 | 0.0370 |
| moderate | 93 | CrowdMAC | 4.60 | 348.5 | 37.36 | 49.6 | 0.4894 |
| moderate | 93 | Holt + last | 14.23 | 142.3 | 15.80 | 15.9 | 0.2731 |
| moderate | 93 | Holt + uniform | 14.23 | 388.9 | 42.12 | 51.9 | 0.7535 |
| dynamic | 87 | Stream-CDF | 5.07 | 81.6 | 7.37 | 22.6 | 0.0820 |
| dynamic | 87 | persistence | 5.73 | 82.1 | 7.95 | 18.5 | 0.0299 |
| dynamic | 87 | CrowdMAC | 5.59 | 339.4 | 36.67 | 46.0 | 0.4944 |
| dynamic | 87 | Holt + last | 13.93 | 137.5 | 15.41 | 14.2 | 0.3378 |
| dynamic | 87 | Holt + uniform | 13.93 | 394.6 | 42.76 | 52.5 | 0.8191 |
| surge | 89 | Stream-CDF | 5.39 | 80.6 | 7.35 | 21.8 | 0.0685 |
| surge | 89 | persistence | 6.02 | 80.2 | 7.73 | 18.4 | 0.0259 |
| surge | 89 | CrowdMAC | 7.44 | 337.3 | 36.84 | 42.6 | 0.5013 |
| surge | 89 | Holt + last | 10.25 | 108.6 | 11.63 | 15.6 | 0.0882 |
| surge | 89 | Holt + uniform | 10.25 | 383.3 | 41.68 | 49.8 | 0.5931 |
| all | 361 | Stream-CDF | 5.34 | 82.3 | 7.53 | 22.1 | 0.0834 |
| all | 361 | persistence | 5.54 | 81.5 | 7.79 | 19.2 | 0.0348 |
| all | 361 | CrowdMAC | 5.59 | 331.5 | 35.70 | 45.9 | 0.4952 |
| all | 361 | Holt + last | 12.52 | 126.6 | 13.98 | 14.7 | 0.4207 |
| all | 361 | Holt + uniform | 12.52 | 367.6 | 39.92 | 48.3 | 0.8905 |

Reconstructed from stored per-clip arrays; no training or full-map transport was rerun.
