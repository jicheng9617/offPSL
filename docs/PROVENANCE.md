# Source and implementation provenance

This repository publishes two methods: OffPSL and a local offline STCH-PSL
adaptation. The compact implementation was written independently for the release,
using the current OffPSL manuscript equations and inspecting the local research
implementation to establish method behavior. No third-party repository source,
benchmark package, dataset, or pretrained model was copied into the method package.
The figure assets include a compact export of all six Fig. 1 methods and PCD's
archived generated results, with source paths and SHA256 hashes, as documented
below. These additional results do not add baseline implementations to the package.

| Release component | Basis and attribution |
|:--|:--|
| Objective ensemble, preference map, training loop | Independent implementation of the current OffPSL method; research reference modules were `cpd_psl/models/sra_psl.py` and `cpd_psl/train/train_sra_psl.py` |
| Denoising support model | Independent single-step denoising MLP and the current manuscript's support/covariance equations; no upstream diffusion implementation |
| Smooth Tchebycheff scalarizer | Mathematical method from Lin et al., ICML 2024; independent implementation with the local offline absolute-distance convention |
| STCH-PSL training | Independent full-data offline adaptation; inspected local `scripts/run_psl_mobo_baselines.py` for the surrogate-plus-map pattern, without copying its other methods or benchmark integrations |
| Array interface, config, CLI, checkpoints, tests | New release code |
| ZDT2 example | Direct analytic ZDT2 equations; generated arrays and actual training/evaluation from a declared seed |
| README figure | Replotted by `examples/plot_fig1.py` from original external observations, all six Fig. 1 methods, and the archived PCD run; recorded objective values are preserved |
| `assets/fig1_results.json` | Lossless export of 256 valid objective rows per method and available preferences; OffPSL/STCH-PSL saved designs are also included; no benchmark observations or baseline source code |

The prior local release had a nearest-neighbor support proxy. That proxy was not
carried forward because it does not implement the current manuscript's denoising
support definition. The original research and earlier release remain separate.

## Original STCH method

Xi Lin, Xiaoyuan Zhang, Zhiyuan Yang, Fei Liu, Zhenkun Wang, and Qingfu Zhang.
*Smooth Tchebycheff Scalarization for Multi-Objective Optimization.* ICML 2024,
PMLR 235:30479–30509.

- [Paper](https://proceedings.mlr.press/v235/lin24y.html)
- [Original authors' code](https://github.com/Xi-L/STCH)

The local code is not the authors' official implementation. Please retain their
method citation when using or adapting STCH-PSL.

## External benchmark

[Off-MOO-Bench / offline-moo](https://github.com/lamda-bbo/offline-moo) remains an
external project. Use its official installation and data instructions and observe
the applicable licenses. It is neither vendored nor required by the core package
or the self-contained example.

The MIT license applies to this release's independent code. It does not relicense
external projects or data.
