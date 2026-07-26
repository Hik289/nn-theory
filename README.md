# Neural Network Theory Experiments

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/license-pending-0E7C66.svg" alt="License"></a> <a href="#paper-or-reference"><img src="https://img.shields.io/badge/paper-reference-1F4E79.svg" alt="Paper or reference"></a> <img src="https://img.shields.io/badge/language-Jupyter%20Notebook-F37626.svg" alt="Jupyter Notebook">
</p>

<p align="center">
  <strong>Notebook-driven experiments for approximation, random features, and training dynamics.</strong>
</p>

<p align="center">
  <img src="assets/readme-figure.png" alt="Neural Network Theory Experiments overview" width="100%">
</p>

## Abstract

This repository is a conference-style artifact for neural-network theory experiments with notebooks and saved figures. It packages the code and notes needed to inspect the central research question: How do approximation, projection pursuit, and random-feature views align empirically? The emphasis is on transparent entry points, reproducible execution, and clear separation between code, local data, and generated outputs.

## Artifact at a Glance

| Item | Details |
| --- | --- |
| Research question | How do approximation, projection pursuit, and random-feature views align empirically? |
| Primary artifact | Theory notebooks, result folders, and figure-generation notebooks. |
| Main entry points | `ppr&nnr.ipynb`, `ppr&nnr_figure*.ipynb`, `result/` |
| Expected outputs | Approximation curves, logs, and report-ready figures. |

## Repository Structure

| Item | Details |
| --- | --- |
| `ppr&nnr.ipynb` and variants | main notebooks for projection-pursuit and neural-network regression experiments. |
| `result/`, `result1/` | saved outputs from experiment runs. |
| `log/` | execution logs and intermediate notes. |
| `ppr&nnr figure*` notebooks | figure-generation notebooks for report plots. |

## Reproducibility Protocol

1. `git clone git@github.com:Hik289/nn-theory.git`
2. `python -m venv .venv && source .venv/bin/activate`
3. `python -m pip install -U pip jupyter numpy scipy matplotlib scikit-learn`
4. Open the main notebook first, then use the figure notebooks to regenerate plots.
5. Record the data window, random seed, software versions, machine type, and exact command used for any full rerun.
6. Store regenerated figures, tables, checkpoints, or reports under the existing result folders instead of overwriting raw inputs.

## Evaluation Protocol

| Step | Reviewer-facing check |
| --- | --- |
| Environment | Confirm the listed runtime or notebook environment starts without modifying tracked files. |
| Minimal run | Execute the smallest entry point before launching longer experiments. |
| Output check | Compare regenerated files with the expected figures, tables, logs, or reports named in this README. |
| Extension check | Add new runs as separate scripts, notebooks, or result folders with explicit names. |

## Expected Results

- The main scripts or notebooks should regenerate the project-specific artifacts listed in **Artifact at a Glance**.
- Outputs should be traceable to a command, parameter setting, and data window.
- Any private data path or machine-specific setting should be documented before sharing the artifact externally.

## Paper or Reference

No external paper link is currently attached to this project. For now, the code, notebooks, and notes in this repository are the primary reference artifact.

## Citation

If this repository supports a paper, cite the paper first and the artifact version second. If no paper is attached, cite the repository snapshot used in the experiment.

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

| Claim | How to inspect it |
| --- | --- |
| Code availability | Code and notebooks are present in the repository. |
| Reproducibility | The protocol above gives the expected setup and run order. |
| Result traceability | Generated outputs should live in named result, report, log, or output folders. |
| Extensibility | New experiments should preserve existing artifacts and add clearly named outputs. |
