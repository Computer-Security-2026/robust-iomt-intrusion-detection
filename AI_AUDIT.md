# AI_AUDIT Team AI Verification Log

**Course:** TBD - the supplied AI audit template says "CS 4379H Cryptography," while the supplied group-project brief says "CS 4371 Computer System Security." Confirm the correct label with the instructor.

**Team name / number:** TBD

**Paper:** *Securing Healthcare with Deep Learning: A CNN-Based Model for Medical IoT Threat Detection*

**AI assistant(s) used:** OpenAI Codex (GPT-5)

## How to use this log

AI use in this project is required. Every AI output the team relies on - including a repeated claim, included citation, committed code, or adopted idea - must appear as one row, whether or not it proved correct. Throwaway drafts and outputs that are discarded without use do not need to be logged.

Keep one log in the repository root and update it as the work progresses. The commit history should show the log growing over the semester. Do not reconstruct it only at the end.

## Categories

| Code | Category |
| --- | --- |
| FC | Fabricated citation: a paper, author, or venue does not exist or does not support the claim |
| NA | Nonexistent API or function: a library, method, flag, or parameter does not exist |
| MR | Misstated result: the source exists, but the number, finding, or claim is wrong |
| CW | Code runs but is wrong: it executes without an error but produces an incorrect result |
| OC | Overclaim: a statement that is true in a narrow case is presented as generally true |
| OK | Verified correct: the output survived verification |

## Log

Allowed roles: Reviewer, Archaeologist, Student Researcher, Reproducibility Checker, and Quanta Correspondent. Add rows as needed.

| # | Date | Role | What the AI claimed or produced | Category | Verified? | How it was checked | Team member |
| ---: | --- | --- | --- | :---: | :---: | --- | --- |
| 1 | 2026-09-29 | Student Researcher | Proposed studying the robustness of the paper's CNN-based IoMT detector when network-flow features are missing or corrupted. | OK | Partially | The idea directly retains the paper's IoMT intrusion-detection task, dataset, and CNN baseline. Experimental feasibility still requires a successful baseline run. | Person 4 - name TBD |
| 2 | 2026-09-29 | Reproducibility Checker | Proposed reproducing binary classification, applying controlled feature corruption, and testing imputation or noise-augmented training. | OK | Partially | The original repository documents a binary configuration and a training command. The corruption and mitigation stages are proposed extensions and remain unverified until implemented. | Person 4 - name TBD |
| 3 | 2026-09-29 | Reproducibility Checker | Claimed that the original repository supports binary, 6-class, and 19-class configurations. | OK | Yes | Checked the original repository README, which documents `python main.py --class_config 2`, `6`, or `19`. | Person 4 - name TBD |
| 4 | 2026-09-29 | Reviewer | Produced the root-level Markdown audit structure and category definitions used in this file. | OK | Yes | Compared it field by field with the instructor-provided `AI_AUDIT_Template_Fall2026.docx`. | Person 4 - name TBD |

## End-of-semester summary

Complete this section before the final demonstration.

- **Total AI outputs logged:** TBD
- **Verified correct:** TBD
- **Failed verification:** TBD
- **Failure 1:** TBD
- **Failure 2:** TBD
- **Verification habit that saved the most time:** TBD
