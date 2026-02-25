"""BERTurk NER inference module for Label Studio ML backend.

Pure inference — no label-studio-ml SDK dependency.
Fixes applied:
  #1  No SDK — plain Flask (see _wsgi.py)
  #2  Orphan I-TERM treated as B-TERM
  #3  Sliding window for texts > MAX_LENGTH tokens
  #4  Labels read from model.config.id2label (not hardcoded)
  #5  Explicit warning on label-config fallback
"""

import os
import re
import logging
import threading
import xml.etree.ElementTree as ET

import json as _json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "")  # resolved in load_model()
MODEL_VERSION = "v018"
MAX_LENGTH = 512
DEFAULT_SCORE_THRESHOLD = 0.70

# ── Global model state (singleton, loaded once) ──────────────────

_tokenizer = None
_model = None
_device = None
_id2label = None  # {0: "O", 1: "B-TERM", 2: "I-TERM"}
_is_dual_head = False  # Whether the loaded model is dual-head
_is_crf = False        # Whether the loaded model uses CRF
_is_bioes = False      # Whether the label set uses BIOES scheme
_infer_lock = threading.Lock()  # HF fast tokenizer is not thread-safe


def _coerce_score_threshold(value, fallback):
    """Convert threshold input to float in [0, 1], else fallback."""
    if value is None:
        return fallback
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid score threshold '%s'; using %.2f", value, fallback)
        return fallback

    if threshold < 0.0 or threshold > 1.0:
        logger.warning(
            "Score threshold %.3f out of range [0,1]; clamping",
            threshold,
        )
    return min(max(threshold, 0.0), 1.0)


DEFAULT_SCORE_THRESHOLD = _coerce_score_threshold(
    os.getenv("NER_SCORE_THRESHOLD"),
    DEFAULT_SCORE_THRESHOLD,
)


def load_model():
    """Load model and tokenizer.  Called once at startup."""
    global _tokenizer, _model, _device, _id2label, _is_dual_head, _is_crf, _is_bioes

    model_dir = _resolve_model_dir()

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model from %s on %s", model_dir, _device)

    _tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Model type detection from config.json
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = _json.load(f)

    _is_crf = cfg.get("crf", False)

    if _is_crf:
        from bert_crf_model import BertCRFForNER
        _model = BertCRFForNER.from_pretrained(model_dir)
        _is_dual_head = False
        logger.info("BERT-CRF model detected")
    elif cfg.get("dual_head", False):
        from dual_head_model import BertForDualHeadNER
        _model = BertForDualHeadNER.from_pretrained(model_dir)
        _is_dual_head = True
        logger.info("Dual-head model detected")
    else:
        _model = AutoModelForTokenClassification.from_pretrained(model_dir)
        _is_dual_head = False

    _model.to(_device)
    _model.eval()

    # Fix #4: dynamic label mapping from model config
    _id2label = _model.config.id2label
    # Detect BIOES scheme from label set
    label_values = set(_id2label.values()) if isinstance(_id2label, dict) else set()
    _is_bioes = "E-TERM" in label_values or "S-TERM" in label_values
    logger.info("Model loaded — labels: %s, dual_head: %s, crf: %s, bioes: %s",
                _id2label, _is_dual_head, _is_crf, _is_bioes)


def _resolve_model_dir():
    """Resolve MODEL_DIR with sensible fallbacks for local / container use."""
    candidates = [
        MODEL_DIR,           # env var or explicit
        "/app/model",        # Docker default
        "./model",           # local relative
    ]
    for path in candidates:
        if path and os.path.isdir(path) and os.path.isfile(
            os.path.join(path, "config.json")
        ):
            return path

    checked = [c for c in candidates if c]
    raise FileNotFoundError(
        f"Model not found. Checked: {checked}. "
        "Set MODEL_DIR env var to the directory containing config.json "
        "and model.safetensors."
    )


# ── Public API ────────────────────────────────────────────────────


def predict_tasks(tasks, label_config=None, score_threshold=None):
    """Run NER on Label Studio tasks.

    Returns list of prediction dicts in Label Studio format::

        [{"result": [...], "score": 0.87, "model_version": "silver_v1"}, ...]
    """
    from_name, to_name, value = _parse_label_config(label_config)
    threshold = _coerce_score_threshold(score_threshold, DEFAULT_SCORE_THRESHOLD)
    if _is_crf and threshold > 0.0:
        logger.warning(
            "CRF model has no per-token confidence — all scores are 1.0. "
            "Score threshold %.2f will have no filtering effect.", threshold
        )

    predictions = []
    for task in tasks:
        text = task["data"].get(value, task["data"].get("text", ""))
        branch = task["data"].get("branch", "")

        # [Branş] prefix — model was trained with this format
        prefix = f"[{branch}] " if branch else ""
        input_text = prefix + text
        prefix_len = len(prefix)

        # Inference (sentence chunking for long texts)
        entities = _predict_single_text(input_text, prefix_len, text)
        filtered_entities = [ent for ent in entities if ent["score"] >= threshold]

        # Format for Label Studio
        results = []
        scores = []
        for ent in filtered_entities:
            results.append(
                {
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "labels",
                    "value": {
                        "start": ent["start"],
                        "end": ent["end"],
                        "text": ent["text"],
                        "labels": ["TERM"],
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
                "model_version": MODEL_VERSION,
                "score_threshold": round(threshold, 4),
            }
        )

    return predictions


# ── Inference internals ───────────────────────────────────────────


def _predict_single_text(input_text, prefix_len, text):
    """Run NER on a single text, chunking into sentences if needed.

    Fix #3: for texts exceeding MAX_LENGTH tokens, splits at sentence
    boundaries and processes each chunk independently.  The model was
    fine-tuned on short sentences, so sentence-level chunks match the
    training distribution and yield far better results than stride-based
    sliding windows.

    Returns list of entity dicts: [{start, end, text, score}, ...]
    """
    with _infer_lock:
        token_count = len(_tokenizer.encode(input_text, add_special_tokens=False))
    usable = MAX_LENGTH - 2

    if token_count <= usable:
        preds, offsets, confs, boundary_probs = _single_window(input_text)
        return _extract_entities(preds, offsets, confs, prefix_len, text, boundary_probs)

    # Long text → sentence-based chunking
    logger.info(
        "Text has %d tokens (limit %d) — splitting into sentence chunks",
        token_count,
        usable,
    )
    return _chunked_predict(input_text, prefix_len, text)


def _single_window(input_text):
    """Run model on a single text that fits within MAX_LENGTH.

    Returns (preds, offsets, confs, boundary_probs) where boundary_probs
    is a (seq_len, 3) tensor if dual-head model, else None.

    Handles standard, dual-head, and CRF model types:
    - Standard: logits → argmax
    - Dual-head: bio_logits → argmax, boundary_logits → boundary_probs
    - CRF: emissions → Viterbi decode (no per-token confidence)
    """
    with _infer_lock:
        enc = _tokenizer(
            input_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(_device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = _model(**inputs)

    boundary_probs = None

    if _is_crf:
        # CRF: Viterbi decode (same logic as model_loader.extract_preds)
        emissions = outputs["logits"]
        seq_total = emissions.size(1)

        # Guard: if only [CLS]+[SEP] or less, return all O
        attn = inputs["attention_mask"][0]
        seq_len = int(attn.sum().item())
        if seq_len <= 2:
            preds = [0] * seq_total
            confs = [1.0] * seq_total
            return preds, offsets, confs, None

        # Slice past [CLS] — torchcrf requires first unmasked timestep
        emissions_nocls = emissions[:, 1:, :]
        crf_mask = attn[1:].bool().clone()
        crf_mask[seq_len - 2] = False  # mask [SEP]

        decoded = _model.decode(emissions_nocls, mask=crf_mask.unsqueeze(0))[0]

        # Map back to full sequence positions
        preds = [0] * seq_total
        mask_positions = crf_mask.nonzero(as_tuple=True)[0].tolist()
        for j, mpos in enumerate(mask_positions):
            if j < len(decoded):
                preds[mpos + 1] = decoded[j]

        # CRF has no per-token softmax confidence; use 1.0 as placeholder
        confs = [1.0] * seq_total
        return preds, offsets, confs, None

    if isinstance(outputs, dict) and "bio_logits" in outputs:
        logits = outputs["bio_logits"]
        boundary_probs = torch.softmax(outputs["boundary_logits"], dim=-1)[0]
    else:
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)[0]
    preds = torch.argmax(probs, dim=-1).tolist()
    confs = torch.max(probs, dim=-1).values.tolist()
    return preds, offsets, confs, boundary_probs


# ── Sentence-based chunking ──────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:–—])\s+")


def _chunked_predict(input_text, prefix_len, text):
    """Split long text into individual sentences and process each one.

    The model was fine-tuned on short sentences, so sentence-level
    inference matches the training distribution and yields the best
    results.  The [Branş] prefix is prepended to EVERY segment so
    the model always has the branch context.

    Oversized sentences (> MAX_LENGTH tokens even alone) are further
    split at clause punctuation (, ; : — –) or word boundaries.
    """
    prefix = input_text[:prefix_len]  # e.g. "[Fizik] " or ""
    text_part = input_text[prefix_len:]  # = text

    # First pass: sentence split
    sentences = _SENT_SPLIT.split(text_part)

    # Second pass: break oversized sentences into smaller segments
    usable = MAX_LENGTH - 2
    segments = []  # [(segment_text, offset_in_text)]
    pos = 0
    for sent in sentences:
        sent_start = text_part.index(sent, pos)
        pos = sent_start + len(sent)

        with _infer_lock:
            tok_count = len(
                _tokenizer.encode(prefix + sent, add_special_tokens=False)
            )
        if tok_count <= usable:
            segments.append((sent, sent_start))
        else:
            for sub_text, sub_offset in _split_oversized(sent):
                segments.append((sub_text, sent_start + sub_offset))

    logger.info("Split into %d segments for individual inference", len(segments))

    # Process each segment
    all_entities = []
    for seg_text, seg_start in segments:
        model_input = prefix + seg_text
        preds, offsets, confs, boundary_probs = _single_window(model_input)
        seg_entities = _extract_entities(preds, offsets, confs, prefix_len, seg_text, boundary_probs)

        for ent in seg_entities:
            final_start = ent["start"] + seg_start
            final_end = ent["end"] + seg_start
            all_entities.append(
                {
                    "start": final_start,
                    "end": final_end,
                    "text": text[final_start:final_end],
                    "score": ent["score"],
                }
            )

    return all_entities


def _split_oversized(segment):
    """Split a sentence that exceeds MAX_LENGTH into smaller segments.

    1. Try clause-level punctuation (, ; : — –)
    2. Fall back to word-boundary splits (~600 chars each)

    Returns list of (sub_text, offset_in_segment) tuples.
    """
    # Try clause-level punctuation
    parts = _CLAUSE_SPLIT.split(segment)
    if len(parts) > 1:
        result = []
        p = 0
        for part in parts:
            part_start = segment.index(part, p)
            p = part_start + len(part)
            result.append((part, part_start))
        return result

    # No clause punctuation → split at word boundaries
    # ~4.5 chars/token for Turkish → 150 tokens ≈ 675 chars
    return _split_at_words(segment, target_chars=600)


def _split_at_words(segment, target_chars=600):
    """Split text at whitespace boundaries into chunks of ~target_chars."""
    if len(segment) <= target_chars:
        return [(segment, 0)]

    result = []
    pos = 0
    while pos < len(segment):
        end = min(pos + target_chars, len(segment))
        if end < len(segment):
            # Prefer breaking at a space
            space = segment.rfind(" ", pos + target_chars // 2, end + 1)
            if space > pos:
                end = space
        chunk = segment[pos:end].strip()
        if chunk:
            # offset = first non-space char position
            leading = len(segment[pos:end]) - len(segment[pos:end].lstrip())
            result.append((chunk, pos + leading))
        pos = end
        # Skip inter-chunk whitespace
        while pos < len(segment) and segment[pos] == " ":
            pos += 1

    return result


# ── Entity extraction ─────────────────────────────────────────────


_WORD_CONNECTORS = {"'", "’", "-"}


def _is_word_char(ch):
    return ch.isalnum() or ch in _WORD_CONNECTORS


def _expand_to_word_boundaries(start, end, text):
    """Expand a predicted span to full word boundaries when inside a word."""
    text_len = len(text)
    start = max(0, min(start, text_len))
    end = max(start, min(end, text_len))

    while start > 0 and start < text_len:
        if _is_word_char(text[start - 1]) and _is_word_char(text[start]):
            start -= 1
            continue
        break

    while end > 0 and end < text_len:
        if _is_word_char(text[end - 1]) and _is_word_char(text[end]):
            end += 1
            continue
        break

    return start, end


def _is_likely_orphan_suffix(start, end, prefix_len, text):
    """Detect short orphan I-TERM pieces that are likely word suffixes."""
    local_start = max(start - prefix_len, 0)
    local_end = max(end - prefix_len, 0)
    if local_end <= local_start:
        return False

    piece = text[local_start:local_end]
    if len(piece) > 4:
        return False

    if not piece.islower():
        return False

    if local_start == 0:
        return False

    return _is_word_char(text[local_start - 1])


def _normalize_entity_spans(spans, text):
    """Normalize spans to word boundaries and merge overlaps/connectors."""
    if not spans:
        return []

    normalized = []
    for span in sorted(spans, key=lambda x: (x["start"], x["end"])):
        start, end = _expand_to_word_boundaries(span["start"], span["end"], text)
        if end <= start:
            continue

        candidate = {
            "start": start,
            "end": end,
            "score": span["score"],
            "text": text[start:end],
        }

        if not normalized:
            normalized.append(candidate)
            continue

        prev = normalized[-1]
        has_connector = (
            candidate["start"] == prev["end"] + 1
            and prev["end"] < len(text)
            and text[prev["end"]] in _WORD_CONNECTORS
        )

        if candidate["start"] <= prev["end"] or has_connector:
            merged_start = min(prev["start"], candidate["start"])
            merged_end = max(prev["end"], candidate["end"])

            prev_len = max(prev["end"] - prev["start"], 1)
            cand_len = max(candidate["end"] - candidate["start"], 1)
            merged_score = (
                (prev["score"] * prev_len) + (candidate["score"] * cand_len)
            ) / (prev_len + cand_len)

            prev["start"] = merged_start
            prev["end"] = merged_end
            prev["score"] = round(merged_score, 4)
            prev["text"] = text[merged_start:merged_end]
        else:
            normalized.append(candidate)

    return normalized


def _extract_entities(preds, offsets, confs, prefix_len, text, boundary_probs=None):
    """Extract NER spans from token-level predictions.

    Supports BIO and BIOES label schemes:
      - B-TERM: entity start
      - I-TERM: entity continuation
      - E-TERM: entity end (BIOES)
      - S-TERM: single-token entity (BIOES)
      - O: outside

    Fix #2: an I-TERM that has no preceding B-TERM is promoted to a new
    entity start (instead of being silently dropped).

    Boundary rescue: when dual-head model provides boundary_probs,
    tokens with BIO=O but high START/END probability can rescue
    missed entities.
    """
    entities = []
    current = None

    for pred_id, (start, end), conf in zip(preds, offsets, confs):
        if start == 0 and end == 0:  # special token ([CLS], [SEP], padding)
            if current:
                entities.append(current)
                current = None
            continue

        label = _id2label.get(pred_id, "O")

        if label == "S-TERM":
            # Single-token entity (BIOES)
            if current:
                entities.append(current)
                current = None
            entities.append({"start": start, "end": end, "scores": [conf]})

        elif label == "B-TERM":
            if current:
                entities.append(current)
            current = {"start": start, "end": end, "scores": [conf]}

        elif label == "I-TERM":
            if current:
                current["end"] = end
                current["scores"].append(conf)
            else:
                if _is_likely_orphan_suffix(start, end, prefix_len, text):
                    logger.debug(
                        "Skipping orphan suffix-like I-TERM at char %d–%d",
                        start,
                        end,
                    )
                    continue
                # Fix #2: orphan I-TERM → promote to entity start
                logger.debug(
                    "Orphan I-TERM at char %d–%d, promoting to entity start",
                    start,
                    end,
                )
                current = {"start": start, "end": end, "scores": [conf]}

        elif label == "E-TERM":
            # Entity end (BIOES)
            if current:
                current["end"] = end
                current["scores"].append(conf)
                entities.append(current)
                current = None
            # else: orphan E-TERM without B/I → ignore

        else:  # O or unknown
            if current:
                entities.append(current)
                current = None

    if current:
        entities.append(current)

    # Adjust character offsets for the [Branş] prefix
    results = []
    for ent in entities:
        adj_start = ent["start"] - prefix_len
        adj_end = ent["end"] - prefix_len
        if adj_end <= 0:
            continue  # entity falls entirely inside prefix
        adj_start = max(adj_start, 0)
        score = sum(ent["scores"]) / len(ent["scores"])
        results.append(
            {
                "start": adj_start,
                "end": adj_end,
                "text": text[adj_start:adj_end],
                "score": round(score, 4),
            }
        )

    # Boundary rescue: recover entities missed by BIO but detected by boundary head
    if boundary_probs is not None and _is_dual_head:
        rescued = _boundary_rescue(preds, offsets, confs, boundary_probs, prefix_len, text, results)
        results.extend(rescued)

    return _normalize_entity_spans(results, text)


def _boundary_rescue(preds, offsets, confs, boundary_probs, prefix_len, text, existing_results):
    """Rescue entities that BIO missed but boundary head detected.

    Strategy: find START tokens (prob > 0.7) followed by END tokens (prob > 0.7)
    where BIO predicted O. Only rescue in regions not already covered by BIO entities.
    Score is discounted by 0.9x to prefer BIO predictions.
    """
    START_THRESH = 0.7
    END_THRESH = 0.7
    SCORE_DISCOUNT = 0.9
    MAX_ENTITY_TOKENS = 15  # max tokens between START and END

    # Build coverage set from existing entities
    covered = set()
    for ent in existing_results:
        for c in range(ent["start"], ent["end"]):
            covered.add(c)

    # Find START/END candidates among O-predicted tokens
    # boundary_probs shape: (seq_len, 3) where 0=O, 1=START, 2=END
    start_candidates = []
    end_candidates = []

    for i, ((s, e), pred_id) in enumerate(zip(offsets, preds)):
        if s == 0 and e == 0:
            continue  # special token
        label = _id2label.get(pred_id, "O")
        if label != "O":
            continue  # only rescue where BIO says O

        start_prob = boundary_probs[i][1].item()
        end_prob = boundary_probs[i][2].item()

        if start_prob > START_THRESH:
            start_candidates.append((i, s, e, start_prob))
        if end_prob > END_THRESH:
            end_candidates.append((i, s, e, end_prob))

    rescued = []
    used_ends = set()

    for s_idx, s_start, s_end, s_prob in start_candidates:
        # Find nearest END after this START
        best_end = None
        for e_idx, e_start, e_end, e_prob in end_candidates:
            if e_idx <= s_idx:
                continue
            if e_idx in used_ends:
                continue
            if e_idx - s_idx > MAX_ENTITY_TOKENS:
                break
            best_end = (e_idx, e_start, e_end, e_prob)
            break

        if best_end is None:
            continue

        e_idx, _, e_end, e_prob = best_end
        adj_start = s_start - prefix_len
        adj_end = e_end - prefix_len

        if adj_end <= 0 or adj_start < 0:
            continue

        # Check if this region overlaps with existing entities
        if any(c in covered for c in range(adj_start, adj_end)):
            continue

        avg_score = (s_prob + e_prob) / 2 * SCORE_DISCOUNT
        entity_text = text[adj_start:adj_end]

        if entity_text.strip():
            rescued.append({
                "start": adj_start,
                "end": adj_end,
                "text": entity_text,
                "score": round(avg_score, 4),
            })
            used_ends.add(e_idx)
            # Mark coverage
            for c in range(adj_start, adj_end):
                covered.add(c)
            logger.debug(
                "Boundary rescue: '%s' at %d-%d (score=%.3f)",
                entity_text, adj_start, adj_end, avg_score,
            )

    if rescued:
        logger.info("Boundary rescue recovered %d entities", len(rescued))

    return rescued


# ── Label-config parsing ─────────────────────────────────────────


def _parse_label_config(label_config_xml):
    """Parse Label Studio XML config → (from_name, to_name, value).

    Fix #5: logs an explicit warning when falling back to defaults,
    so mismatches don't go unnoticed.
    """
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
                logger.info(
                    "Label config parsed: from_name=%s, to_name=%s, value=%s",
                    from_name,
                    to_name,
                    value,
                )
                return from_name, to_name, value

            logger.warning("No <Labels> tag found in label_config XML")
        except ET.ParseError as exc:
            logger.warning("Failed to parse label_config XML: %s", exc)

    # Fallback with loud warning (Fix #5)
    from_name = os.getenv("LABEL_FROM_NAME", "label")
    to_name = os.getenv("LABEL_TO_NAME", "text")
    value = "text"
    logger.warning(
        "USING DEFAULT label config: from_name=%s, to_name=%s, value=%s  — "
        "predictions may be mismatched if your Label Studio project "
        "uses different names.",
        from_name,
        to_name,
        value,
    )
    return from_name, to_name, value
