# Robust IoMT Intrusion Detection

This repository contains the CS 4371 group project based on **Securing Healthcare with Deep Learning: A CNN-Based Model for Medical IoT Threat Detection**. The project will reproduce a manageable baseline from the published implementation and evaluate how the detector behaves when network-flow features are incomplete or noisy.

## Research question

How robust is a CNN-based medical Internet of Things intrusion detector when selected network-flow features are missing or corrupted?

## Proposed contribution

The project will:

1. reproduce the paper's binary benign-versus-malicious classification baseline;
2. evaluate the baseline with clean test data;
3. introduce controlled feature corruption at several levels;
4. measure accuracy, macro-F1, false-negative rate, and confusion matrices; and
5. evaluate a simple mitigation such as median imputation or noise-augmented training.

The final demonstration is intended to classify sample network-flow records and compare clean-data performance with performance under feature corruption. The precise corruption method and levels will be finalized after the baseline runs successfully.

## Source material

- Paper: *Securing Healthcare with Deep Learning: A CNN-Based Model for Medical IoT Threat Detection*
- Original implementation: https://github.com/alirezamohamadiam/Securing-Healthcare-with-Deep-Learning-A-CNN-Based-Model-for-medical-IoT-Threat-Detection
- Dataset referenced by the implementation: CICIoMT2024

## Team responsibilities

| Responsibility | Owner | Deliverable |
| --- | --- | --- |
| Independent reviewer | TBD | One-page independent paper review completed before reading the instructor AI review |
| Student researcher | TBD | Proposed topic, course connection, applications, and AI Idea Appendix |
| Reproducibility checker | TBD | Baseline, robustness experiment, demo, and AI-code verification plan |
| Audit and integration | TBD | AI disagreement table, `AI_AUDIT.md`, repository organization, and final proposal assembly |

## Repository layout

```text
.
|-- AI_AUDIT.md        # Required team AI verification log
|-- README.md           # Project overview and status
|-- data/README.md      # Dataset acquisition and handling policy
|-- proposal/README.md  # Proposal assembly checklist
|-- src/                # Implementation code (added after proposal approval)
|-- tests/              # Automated checks
`-- results/            # Reproducible metrics and figures
```

## Reproducibility rules

- Do not commit the complete dataset or generated model artifacts unless their licenses and sizes permit it.
- Record the exact dataset files, preprocessing choices, split strategy, random seeds, package versions, and commands used for every reported result.
- Compare performance with macro-F1 and false-negative rate in addition to accuracy.
- Treat an AI-generated code suggestion as unverified until it has been reviewed and tested.
- Record every AI output the team relies on in [`AI_AUDIT.md`](AI_AUDIT.md).

## Current status

- [x] Repository created
- [x] Project topic selected
- [x] AI verification log initialized
- [ ] Course identifier confirmed with the instructor
- [ ] Team names and responsibilities added
- [ ] Independent review completed and locked
- [ ] Instructor AI review comparison completed
- [ ] Proposal submitted
- [ ] Baseline reproduced
- [ ] Robustness experiment implemented
