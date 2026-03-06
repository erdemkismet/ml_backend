"""Generic NER inference module (PER / LOC / ORG) for Label Studio ML backend.

Uses a standard BertForTokenClassification model with BIO tagging.
No CRF, dual-head, BIOES, or branch-prefix logic — kept intentionally simple.
"""

import logging
import os
import re
import threading
import xml.etree.ElementTree as ET

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

logger = logging.getLogger(__name__)

NER_MODEL_DIR = os.getenv("NER_MODEL_DIR", "")
NER_MODEL_VERSION = "v001"
NER_HF_REPO_ID = os.getenv("NER_HF_REPO_ID", "akdeniz27/bert-base-turkish-cased-ner")
MAX_LENGTH = 512
DEFAULT_SCORE_THRESHOLD = 0.50

# ── Global model state (singleton, loaded once) ──────────────────

_tokenizer = None
_model = None
_device = None
_id2label = None
_infer_lock = threading.Lock()


def load_ner_model():
    """Load NER model and tokenizer.  Called once at startup."""
    global _tokenizer, _model, _device, _id2label

    model_dir = _resolve_ner_model_dir()

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading NER model from %s on %s", model_dir, _device)

    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForTokenClassification.from_pretrained(model_dir)
    _model.to(_device)
    _model.eval()

    _id2label = _model.config.id2label
    logger.info("NER model loaded — labels: %s", _id2label)


def _resolve_ner_model_dir():
    """Resolve NER_MODEL_DIR with fallbacks.  Downloads from HF if needed."""
    candidates = [
        NER_MODEL_DIR,
        "/app/ner_model",
        "./ner_model",
    ]
    for path in candidates:
        if path and os.path.isdir(path) and os.path.isfile(
            os.path.join(path, "config.json")
        ):
            return path

    target_dir = NER_MODEL_DIR or "/app/ner_model"
    logger.info(
        "NER model not found locally. Downloading from HF: %s → %s",
        NER_HF_REPO_ID,
        target_dir,
    )
    try:
        from huggingface_hub import snapshot_download

        token = os.getenv("HF_TOKEN") or None
        snapshot_download(NER_HF_REPO_ID, local_dir=target_dir, token=token)
        logger.info("NER model downloaded successfully to %s", target_dir)
        return target_dir
    except Exception as exc:
        checked = [c for c in candidates if c]
        raise FileNotFoundError(
            f"NER model not found locally ({checked}) and HF download failed: {exc}. "
            "Set NER_MODEL_DIR or NER_HF_REPO_ID environment variables."
        ) from exc


# ── Public API ────────────────────────────────────────────────────


def predict_tasks(tasks, label_config=None, score_threshold=None):
    """Run NER on Label Studio tasks.

    Returns list of prediction dicts in Label Studio format.
    """
    from_name, to_name, value = _parse_label_config(label_config)
    threshold = _coerce_threshold(score_threshold, DEFAULT_SCORE_THRESHOLD)

    predictions = []
    for task in tasks:
        text = task["data"].get(value, task["data"].get("text", ""))
        entities = _predict_single_text(text)
        filtered = [e for e in entities if e["score"] >= threshold]

        results = []
        scores = []
        for ent in filtered:
            results.append(
                {
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "labels",
                    "value": {
                        "start": ent["start"],
                        "end": ent["end"],
                        "text": ent["text"],
                        "labels": [ent["label"]],
                    },
                    "score": ent["score"],
                }
            )
            scores.append(ent["score"])

        avg_score = sum(scores) / len(scores) if scores else 0.0
        predictions.append(
            {
                "result": results,
                "score": round(avg_score, 4),
                "model_version": NER_MODEL_VERSION,
                "score_threshold": round(threshold, 4),
            }
        )

    return predictions


# ── Inference internals ───────────────────────────────────────────


def _predict_single_text(text):
    """Run NER on a single text, chunking into sentences if needed."""
    with _infer_lock:
        token_count = len(_tokenizer.encode(text, add_special_tokens=False))
    usable = MAX_LENGTH - 2

    if token_count <= usable:
        preds, offsets, confs = _single_window(text)
        return _extract_entities_bio(preds, offsets, confs, text)

    logger.info(
        "NER text has %d tokens (limit %d) — splitting into sentences",
        token_count,
        usable,
    )
    return _chunked_predict(text)


def _single_window(text):
    """Run model on a single text that fits within MAX_LENGTH."""
    with _infer_lock:
        enc = _tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(_device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    preds = torch.argmax(probs, dim=-1).tolist()
    confs = torch.max(probs, dim=-1).values.tolist()
    return preds, offsets, confs


_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")


def _chunked_predict(text):
    """Split long text into sentences and process each one."""
    sentences = _SENT_SPLIT.split(text)

    usable = MAX_LENGTH - 2
    segments = []  # (segment_text, offset_in_text)
    pos = 0
    for sent in sentences:
        sent_start = text.index(sent, pos)
        pos = sent_start + len(sent)

        with _infer_lock:
            tok_count = len(_tokenizer.encode(sent, add_special_tokens=False))

        if tok_count <= usable:
            segments.append((sent, sent_start))
        else:
            # Oversized sentence → word-boundary split
            for sub_text, sub_offset in _split_at_words(sent):
                segments.append((sub_text, sent_start + sub_offset))

    logger.info("NER split into %d segments", len(segments))

    all_entities = []
    for seg_text, seg_start in segments:
        preds, offsets, confs = _single_window(seg_text)
        seg_entities = _extract_entities_bio(preds, offsets, confs, seg_text)

        for ent in seg_entities:
            all_entities.append(
                {
                    "start": ent["start"] + seg_start,
                    "end": ent["end"] + seg_start,
                    "text": text[ent["start"] + seg_start : ent["end"] + seg_start],
                    "label": ent["label"],
                    "score": ent["score"],
                }
            )

    return all_entities


def _split_at_words(segment, target_chars=600):
    """Split text at whitespace boundaries into chunks of ~target_chars."""
    if len(segment) <= target_chars:
        return [(segment, 0)]

    result = []
    pos = 0
    while pos < len(segment):
        end = min(pos + target_chars, len(segment))
        if end < len(segment):
            space = segment.rfind(" ", pos + target_chars // 2, end + 1)
            if space > pos:
                end = space
        chunk = segment[pos:end].strip()
        if chunk:
            leading = len(segment[pos:end]) - len(segment[pos:end].lstrip())
            result.append((chunk, pos + leading))
        pos = end
        while pos < len(segment) and segment[pos] == " ":
            pos += 1

    return result


# ── Entity extraction (BIO) ──────────────────────────────────────


def _extract_entities_bio(preds, offsets, confs, text):
    """Extract NER spans from BIO-tagged token predictions.

    Labels are dynamic (read from _id2label).  Handles:
      B-{TYPE}: entity start
      I-{TYPE}: entity continuation (must match current type)
      O: outside
    Orphan I-{TYPE} without preceding B → promoted to entity start.
    """
    entities = []
    current = None  # {"start", "end", "label", "scores"}

    for pred_id, (start, end), conf in zip(preds, offsets, confs):
        if start == 0 and end == 0:  # special token
            if current:
                entities.append(current)
                current = None
            continue

        label = _id2label.get(pred_id, "O") if _id2label else "O"

        if label.startswith("B-"):
            if current:
                entities.append(current)
            entity_type = label[2:]
            current = {"start": start, "end": end, "label": entity_type, "scores": [conf]}

        elif label.startswith("I-"):
            entity_type = label[2:]
            if current and current["label"] == entity_type:
                current["end"] = end
                current["scores"].append(conf)
            else:
                # Orphan I- or type mismatch → promote to new entity
                if current:
                    entities.append(current)
                current = {"start": start, "end": end, "label": entity_type, "scores": [conf]}

        else:  # O
            if current:
                entities.append(current)
                current = None

    if current:
        entities.append(current)

    # Build final results
    results = []
    for ent in entities:
        s, e = ent["start"], ent["end"]
        if e <= s:
            continue
        score = sum(ent["scores"]) / len(ent["scores"])
        results.append(
            {
                "start": s,
                "end": e,
                "text": text[s:e],
                "label": ent["label"],
                "score": round(score, 4),
            }
        )

    return results


# ── Label-config parsing ─────────────────────────────────────────


def _parse_label_config(label_config_xml):
    """Parse Label Studio XML config → (from_name, to_name, value)."""
    if label_config_xml:
        try:
            root = ET.fromstring(label_config_xml)
            labels_tag = root.find(".//Labels")
            if labels_tag is not None:
                from_name = labels_tag.get("name")
                to_name = labels_tag.get("toName")
                text_tag = root.find(f'.//Text[@name="{to_name}"]')
                value = (
                    text_tag.get("value", "").lstrip("$")
                    if text_tag
                    else to_name
                )
                return from_name, to_name, value
        except ET.ParseError as exc:
            logger.warning("NER: Failed to parse label_config XML: %s", exc)

    return "label", "text", "text"


def _coerce_threshold(value, fallback):
    """Convert threshold to float in [0, 1], else fallback."""
    if value is None:
        return fallback
    try:
        t = float(value)
    except (TypeError, ValueError):
        return fallback
    return min(max(t, 0.0), 1.0)
