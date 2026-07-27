# Neural Network Theory Experiments

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/license-pending-0E7C66.svg" alt="License"></a> <a href="#paper-or-reference"><img src="https://img.shields.io/badge/paper-reference-1F4E79.svg" alt="Paper or reference"></a> <img src="https://img.shields.io/badge/language-Jupyter%20Notebook-F37626.svg" alt="Jupyter Notebook">
</p>

<p align="center">
  <strong>Conference-style artifact for approximation and training-dynamics notebooks.</strong>
</p>

<p align="center">
  <img src="assets/readme-figure.png" alt="Neural Network Theory Experiments overview" width="100%">
</p>

## Abstract

This repository is organized as a conference-style artifact for neural-network theory experiments. It is written for a reviewer or collaborator who wants to identify the exact entry points, understand the expected outputs, and reproduce the core evidence without reverse-engineering the folder layout. The central question is: **How do approximation, projection-pursuit, and random-feature views align empirically?**

## Contribution Summary

- Theory notebooks with saved outputs.
- Result and log folders.
- Figure-generation notebooks for report plots.

## Artifact at a Glance

| Item | Details |
| --- | --- |
| Research question | How do approximation, projection-pursuit, and random-feature views align empirically? |
| Primary contribution | Theory notebooks with saved outputs; Result and log folders; Figure-generation notebooks for report plots |
| Main entry points | `ppr&nnr.ipynb`, `ppr&nnr_figure*.ipynb`, `result/`, `log/` |
| Runtime | Jupyter/Python with NumPy, SciPy, Matplotlib, and scikit-learn |
| Data expectation | Notebook-defined experiments and local generated outputs |
| Expected evidence | Approximation curves, logs, and report-ready figures |

## Repository Structure

| Item | Details |
| --- | --- |
| Entry points | `ppr&nnr.ipynb`, `ppr&nnr_figure*.ipynb`, `result/`, `log/` |
| Experiment assets | Notebook-defined experiments and local generated outputs |
| Generated artifacts | Approximation curves, logs, and report-ready figures |
| Documentation role | README records the reproducibility protocol, reviewer-facing checks, and citation metadata |

## Reproducibility Protocol

1. Clone the repository: `git clone git@github.com:Hik289/nn-theory.git`.
2. Prepare the runtime listed in **Artifact at a Glance**.
3. Start from the main entry points rather than auxiliary folders.
4. Run the smallest script or notebook first to verify local paths and package versions.
5. Record the command, data window, random seed, machine type, and software versions for each full run.
6. Store regenerated figures, logs, tables, checkpoints, or reports in named output folders so the original artifacts remain inspectable.

## Evaluation Protocol

| Check | Expected reviewer action |
| --- | --- |
| Entry-point clarity | Confirm the listed scripts or notebooks are the natural starting points. |
| Minimal execution | Run a small case before attempting the full experiment. |
| Output traceability | Map every regenerated output back to a command and data setting. |
| Result inspection | Compare generated artifacts with the expected evidence listed above. |
| Extension hygiene | Add new experiments as clearly named scripts, notebooks, or output folders. |

## Expected Results

A successful reproduction should produce or refresh the following evidence: Approximation curves, logs, and report-ready figures. If local datasets or machine-specific paths are required, document those paths outside the committed code before sharing the artifact.

## Known Limitations

- Large datasets, private data paths, and machine-specific settings may need local configuration.
- Some historical notebooks or scripts may reflect exploratory runs; prefer the entry points listed above for review.
- For archival release, add a pinned environment file and a small public fixture when possible.

## Paper or Reference

No external paper link is currently attached to this project. Cite the repository snapshot when using the artifact in academic work.

## Citation

If a paper is attached, cite the paper first and this artifact second. Otherwise cite the repository snapshot used for the experiment.

```bibtex
@misc{nn_theory_artifact_2026,
  title = {{Neural Network Theory Experiments}},
  author = {Hik289},
  year = {2026},
  howpublished = {\url{https://github.com/Hik289/nn-theory}},
  note = {Conference-style research artifact}
}
```

## License

No explicit license file is included yet. Add one before public reuse, redistribution, or package release.

## Reviewer Checklist

| Claim | Inspection path |
| --- | --- |
| Code availability | Core scripts, notebooks, and utilities are tracked in this repository. |
| Reproducibility | The protocol above states setup, entry points, and output expectations. |
| Data transparency | Local or private data dependencies should be documented before external release. |
| Result traceability | Generated outputs should live in named result, report, log, checkpoint, or output folders. |
