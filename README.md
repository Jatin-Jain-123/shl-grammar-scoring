# Grammar Scoring Engine for Spoken English

Predicts a 0–5 grammar score (MOS Likert, human-rated) from a 45–60 s audio clip.
SHL Research Engineer assessment, Kaggle competition `shl-hiring-assessment-2026`.

## Results (5-fold stratified cross-validation × 3 repeats, 732 speech clips)

| Model | CV RMSE | CV Pearson |
|---|---|---|
| Always predict the mean | 1.014 | – |
| Fluency features + Whisper text embedding (Ridge) | 0.697 | 0.728 |
| 45 hand-built features only (LightGBM) | 0.682 | 0.741 |
| **Simple model: one SVR on WavLM-large + Whisper-encoder embeddings** | 0.507 | 0.869 |
| **Final: features + 6 embedding blocks, stacked, LightGBM + Ridge average** | **0.488 ± 0.025** | **0.877 ± 0.014** |

**Training RMSE (final model): 0.364** (fit and scored on the same data, so optimistic by design).

### Iteration log

| Version | Change | CV RMSE |
|---|---|---|
| v1 | Whisper + wav2vec2 transcripts, 43 rubric-linked features, WavLM-base + text embeddings | 0.537 |
| v2 | + WavLM-large and Whisper-large-v3 encoder embeddings | 0.513 |
| v3 | + LLM examiner feature; SVR instead of Ridge per embedding | 0.495 |
| v4 | + 10-second window embeddings with a window-level SVR | **0.488** |

## Approach

1. **Audio quality check.** 37 training clips (all labelled 0, none in the test set) are pure static, identified by high spectral flatness *and* low SNR together. They are excluded from training and evaluation; including them would inflate Pearson without improving grammar prediction.
2. **Light preprocessing.** 16 kHz mono, 70 Hz high-pass, loudness normalisation; silence handled by voice-activity detection during transcription so pauses stay measurable.
3. **Speech to text, chosen by experiment.** Whisper large-v3 with a verbatim-style prompt (keeps fillers and real grammar errors) is the main transcript. wav2vec2 (CTC) garbles accented speech, so its non-word rate and its disagreement with Whisper serve as intelligibility features.
4. **Hand-built features (45, grouped by rubric criterion).**
   - A. Fluency and ASR confidence: speaking rate, pauses, fillers, restarts, word confidence.
   - B. Syntax and vocabulary (spaCy): sentence length, parse-tree depth, subordinate clauses, tense variety, word rarity.
   - C. Grammar models: grammatical-error-correction edit rate (T5), CoLA acceptability (RoBERTa), GPT-2 surprise.
   - F. LLM examiner: Qwen2.5-7B-Instruct (4-bit, local) reads SHL's rubric and the transcript; the expected value of its 1–5 digit probabilities is the feature.
5. **Embeddings (frozen pretrained models).** Text: all-mpnet-base-v2 on the Whisper and wav2vec2 transcripts. Audio: WavLM-base-plus; WavLM-large (layers 16–24 averaged); Whisper-large-v3 encoder (layers 28–32 averaged); and the latter two again over 10-second windows (about 9 per clip).
6. **Model.** Each embedding block is condensed into one out-of-fold SVR prediction (stacking; window blocks are averaged per clip). These plus the 45 features feed LightGBM and Ridge, whose predictions are averaged. The whole pipeline is refit inside every CV fold (no leakage).

## Key findings

- **Audio carries most of the signal.** Hand-built features alone reach 0.68; adding audio embeddings brings the model to 0.49. Human grammar ratings are strongly entangled with overall spoken proficiency (pronunciation, fluency, prosody).
- **A simple model gets most of the way:** one SVR on two audio embeddings reaches 0.507.
- **What did not help:** the LLM examiner (its grades cluster at 3–4; r = 0.54 with the human score), and window-level training on its own (it helps only inside the ensemble).
- **Checked and ruled out:** same-numbered train/test files are different speakers, so there is no speaker leak.
- **Largest errors:** polished, memorised-sounding answers (over-scored; the rubric penalises "memorized sentence patterns") and frequent self-corrections (under-scored, although the rubric rewards them).

Full report, visualisations (label distribution, audio QC, layer curves, results, residuals, ablation, SHAP) and limitations are inside the notebook.

## Files

- `grammar_scoring_engine.ipynb`: executed notebook with code, report and all outputs.
- `outputs/submission.csv`: final test-set predictions (216 rows, `filename,label`).
- `outputs/submission_simple.csv`: predictions of the simple one-SVR model.
- `outputs/cv_results.csv`, `outputs/ablation.csv`: evaluation tables.
- `requirements.txt`: extra packages beyond the Kaggle image.

## How to run

Kaggle notebook with GPU T4 and Internet on; attach the competition data; *Save & Run All*.
From scratch it takes about 4 hours (transcription about 70 minutes, audio embeddings about 30 minutes, the LLM about 12 minutes, model training about 1 hour). Each expensive step caches its result as a `feat_*.pkl` / `transcripts_*.jsonl` file; attaching earlier outputs as a Kaggle dataset skips those steps.
