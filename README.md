# ⚖️ Legal Reasoning Auditor (LRA)

**An AI assistant that reads Indian court judgments, flags potential reversible errors, and drafts a first-pass "Grounds of Appeal."**

LRA is built as an *assistive triage tool*: it surfaces issues for a lawyer to review, not a verdict to trust blindly. Every flag is meant to be verified by a human.

---

## The Problem

Appellate lawyers, legal-aid workers, and solo practitioners spend hours reading long judgments line by line to find the specific errors worth appealing — a factual contradiction, a miscited statute, an award figure that doesn't match the record. That work is slow, easy to get wrong under time pressure, and out of reach for those who can't afford large teams.

LRA does the first pass in seconds, so the human can spend their time deciding, not searching.

---

## How It Works

The pipeline turns a raw judgment PDF into a structured, checkable set of findings:

```
   PDF judgment
        │
        ▼
 ┌─────────────────┐   Splits the judgment into its rhetorical parts:
 │  1. Segment     │   Facts · Issues · Arguments · Reasoning · Decision
 │     (RRL model) │   
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐   Checks the court's Reasoning and Decision against
 │  2. Audit       │   the record for contradictions, unsupported claims,
 │                 │   and out-of-record evidence.
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐   Verifies every statute cited against a knowledge base
 │  3. Verify law  │   (e.g. flags repealed IPC sections cited after the 2024
 │     (RAG)       │   BNS transition, or offence–ingredient mismatches).
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐   Produces a structured, template-grounded draft —
 │  4. Draft       │   no fabricated legal claims by construction.
 │   Grounds       │
 └─────────────────┘
        │
        ▼
   Grounds of Appeal
```

---

## What Makes It Work

**Segmentation (Rhetorical Role Labelling).**
A fine-tuned **InLegalBERT → BiLSTM → CRF** model labels each sentence by its role in the judgment (Facts, Reasoning, Decision, and so on), reaching **0.80 validation macro-F1**. This structure is what lets the auditor focus only on the parts that matter — you can't check a court's reasoning until you know which sentences *are* the reasoning.

**Auditing with swappable backends.**
The logical auditor runs on three interchangeable engines — a pairwise NLI model, a local LLM, and an API model — so accuracy and cost can be traded off freely. Careful scope-narrowing (auditing the Decision against the record, with corroboration checks) cut false alarms on clean judgments by **16×**.

**Statute verification.**
A retrieval-grounded checker cross-references cited sections against a curated statute knowledge base, catching miscitations and post-repeal references automatically.

**A benchmark that didn't exist before.**
Because there's no public dataset of "judgments with known errors," LRA ships its own **error-injection benchmark**: it plants realistic mistakes (polarity flips, corrupted figures, fake evidence, statutory miscites, and subtler LLM-generated errors like burden-shifting) into clean judgments, then measures how many the auditor catches — with precision, recall, per-error-type recall, false-flag rates, and confidence intervals.

---

## Results

Evaluated on **712 Supreme Court judgments**.

| Metric | Result |
|---|---|
| Segmentation model (val macro-F1) | **0.80** |
| Auditor F1 (LLM backend) | **0.66** |
| Auditor recall | **0.77** |
| False-alarm reduction (after tuning) | **16×** |

**A real catch.** On an actual judgment, LRA flagged that the court awarded **₹15,402/acre** while the record it cited stated **₹7,100/acre** — quoting the conflicting sentence directly. This is the kind of concrete, verifiable error the tool is built to surface.

---

## Who It's For

- **Appellate lawyers & associates** — a fast first-pass triage of where a judgment may be vulnerable.
- **Legal-aid & solo practitioners** — the biggest impact: the reviewing power of a large team, without one.
- **Law students & clinics** — an explanatory mode that walks through *why* something was flagged, as a teaching aid.

---

## Honest Limitations

LRA is deliberately scoped and transparent about what it can't do:

- **Assistive, not autonomous.** A lawyer must verify every flag. It is a triage aid, not a decision-maker.
- **Catches structural and textual errors** — contradictions, miscitations, out-of-record claims — **not** judgment calls like the weighing of evidence or the application of legal standards.
- **Tested on Supreme Court judgments.** Generalisation to trial-court documents is future work.
- **The statute knowledge base is a curated starter set**, not the full Bare Acts.
- Judgments without a clearly identifiable Decision section can't be audited.

---

## Quick Start

```bash
# Run the full pipeline on a judgment PDF
python scripts/demo_e2e.py \
    --pdf data/sample_judgment.pdf \
    --auditor llm \
    --rrl live
```

- `--auditor` selects the audit backend: `nli` · `llm` · `api`
- `--rrl live` loads the trained segmentation model

---

## Project Structure

```
lra-rrl/     Rhetorical Role Labelling — the segmentation model & training
lra-audit/   The auditor — ingestion, logical audit, statute check, drafting
data/        Judgment samples, statute knowledge base, benchmark sets
scripts/     End-to-end demo, training, and evaluation entry points
```

---

<sub>Built as a capstone project. Designed for the Indian legal system, with the transition from the IPC to the Bharatiya Nyaya Sanhita (BNS) built in.</sub>
