# Synthetic ZDT2 smoke example

This is the self-contained quick-start example, **not the README's main figure**.
Its near-front-biased data deliberately differ from the original benchmark data.
The README now shows archived manuscript Fig. 1 results; see [figure instructions](FIGURE.md).

## Protocol

- Seed: **42**, shared by data generation and each method's initialization.
- Dataset: **4,096 designs**, **6 continuous variables** in `[0,1]`, two minimized objectives.
- Generation: sample uniform designs, independently mark each row with probability
  0.75, and multiply its final five coordinates by `0.2 * U³`, where `U` is uniform
  per row. Unmarked rows retain full-box exploration. This deliberately provides
  both low-tail observations and broader coverage for a compact visual example.
- No exact Pareto solutions are inserted, and there is no objective-based filtering.
- ZDT2: `f1=x1`, `g=1+9*mean(x2,...,x6)`, `f2=g-x1²/g`.
- The analytic front is `(t, 1-t²)` for `t` in `[0,1]`; it is used only for plotting.
- Each method sees exactly the same fixed offline data. Both have the same map
  architecture and map update budget. OffPSL additionally fits five bootstrapped
  surrogates and its denoiser, so total training compute is different.
- Query: **101 preferences**, with first-coordinate weights from 0.01 to 0.99.
- Evaluation: call the analytic objective only after each map is trained. Every
  colored point is a true outcome, not a surrogate prediction.

This is a single-seed mechanism illustration, not a benchmark ranking or a
reproduction of the manuscript's Fig. 1 or reported statistics. The six-dimensional
setup and mixture above are fully specified in the example source. The figure
shows every generated candidate; no favorable candidates are selected.

In this recorded run, OffPSL spans much of the analytic front, while the STCH-PSL
map returns nearly the same design for all queries. Their overlap is labeled in
the plot. This observation is specific to the stated seed, data distribution, and
configuration; the example does not estimate performance across seeds or tasks.

## Regenerate

```bash
python -m pip install -e ".[demo]"
python examples/zdt2.py --seed 42 --device cpu --threads 2 --out runs/zdt2-seed42
```

The default config is [`configs/zdt2.json`](../configs/zdt2.json). CPU execution was
validated with Python 3.12, PyTorch 2.7.1, NumPy 2.1.2 and Matplotlib 3.10.3. Training
took about 30 seconds across both methods in the recorded run, using two PyTorch
threads on an Intel Core i5-14600KF. Package import, plotting, installation and
hardware differences add overhead. Seeds reproduce the recipe; bitwise agreement
across different devices, libraries, or hardware is not promised.

The compact [run record](../assets/synthetic_demo_run.json) contains measured timings,
configurations, environment, data hash, and candidate diagnostics. The complete run
also writes both model checkpoints, full training histories, generated data, and
candidate arrays. These can be regenerated and are not bundled as repository data.
To redraw the synthetic example from those arrays, use `--plot-only`. Its images
remain in the selected run directory, separate from the README's benchmark figure.

## Read the figure

The full offline objective distribution is shown in (a). Panels (b–c) share the
same zoom around the returned objective values and front; their zoom is marked in
(a). Gray observations outside this range remain visible in (a). The common
viridis scale encodes the actual queried preference, rather than a fitted rank.

Panel (d) shows how `x1` changes with preference. Panel (e) shows the mean remaining
coordinates; the ZDT2 Pareto set has all of them zero. The separate method markers,
line styles, labels, and dark front keep the plot readable in grayscale. A static
PNG plus editable SVG is sufficient: each preference is visible simultaneously,
with no animation or player needed to inspect the mapping.
