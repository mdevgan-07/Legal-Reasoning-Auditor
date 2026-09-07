# =====================================================================
# RUN THIS IN JUPYTERLAB — exports the sentence TEXTS (and optionally
# per-sentence confidences) aligned 1:1 with finetuned_bert_predictions.json
#
# The auditor runs NLI on sentence strings, so label-ids alone are not enough.
# You do NOT need the model weights (tar file) for this — only the dataset
# that produced the predictions, in the SAME order.
# =====================================================================
import json

# ---------------------------------------------------------------------
# CASE A — your test dataset exposes raw sentences per document.
# Adjust `test_dataset` / attribute names to match your code. The ONLY
# requirement: iterate documents in the SAME order as test_pred, and for
# each doc emit its list of raw sentence strings (pre-tokenization).
# ---------------------------------------------------------------------
test_sentences = []
for doc in test_dataset:              # <-- your test set object
    # doc should give you the original sentences for that judgment.
    # Common patterns — pick whichever matches your Dataset.__getitem__:
    #   sents = doc["sentences"]
    #   sents = doc.raw_sentences
    #   sents = doc["text"]           # if already a list[str]
    sents = doc["sentences"]          # <-- EDIT to your field name
    test_sentences.append([str(s).strip() for s in sents])

with open("test_sentences.json", "w") as f:
    json.dump(test_sentences, f, ensure_ascii=False)

# sanity: lengths must match the prediction file, doc for doc
pred = json.load(open("finetuned_bert_predictions.json"))["test_pred"]
assert len(pred) == len(test_sentences), (len(pred), len(test_sentences))
for i, (p, s) in enumerate(zip(pred, test_sentences)):
    # predictions may be padded to 200; texts define the true length
    assert len(s) <= len(p), f"doc {i}: {len(s)} texts > {len(p)} labels"
print("OK:", len(test_sentences), "docs exported to test_sentences.json")


# ---------------------------------------------------------------------
# CASE B (OPTIONAL but recommended) — also export per-sentence confidence.
# If your CRF/BiLSTM forward exposes emission or marginal probabilities,
# dump the max-prob per sentence so the auditor can skip low-confidence
# Decision/Issue sentences instead of cascading RRL errors into flags.
#
# If you only have hard Viterbi labels with no probabilities, SKIP this —
# the adapter defaults confidence to 1.0.
# ---------------------------------------------------------------------
# test_confidences = []
# for doc in test_dataset:
#     probs = model.predict_marginals(doc)   # <-- your API; [n_sent, n_labels]
#     test_confidences.append([float(max(row)) for row in probs])
# json.dump(test_confidences, open("test_confidences.json", "w"))
# print("confidences exported")
