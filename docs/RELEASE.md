# Remaining release artifacts

The repository is suitable for a private research handoff. Before a complete
public release, the authors still need to provide or resolve:

- Remaining manuscript assets: `arxiv.sty`, `orcid.pdf`, and figures referenced
  by `paper.tex` other than the included `figs/Problem.png`.
- A reuse license for the original code, paper, and derived prediction artifacts.
- Dataset acquisition instructions with the exact GCS block split manifest.
- Approved checkpoint distribution links, if trained weights will be shared.
- A captured original package/CUDA environment, if available.
- Raw inputs or a documented regeneration path for the kappa sweep and figures.

The bundled predictions reproduce table aggregation. Their presence does not
mean training, full-map evaluation, or the paper PDF has been rerun successfully.
