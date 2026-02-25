"""
Unified NER Model Loader — Tek giriş noktası.

config.json'dan model tipini algılayarak doğru sınıfı yükler.
Tüm inference/eval scriptleri bu modülü kullanmalıdır.
"""

import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

logger = logging.getLogger(__name__)


def load_ner_model(model_dir: str, device: str = None):
    """Tek giriş noktası: config.json'dan model tipini algıla ve yükle.

    Args:
        model_dir: Model dizini (config.json, model weights, tokenizer içerir).
        device: Kullanılacak cihaz. None ise otomatik algılanır.

    Returns:
        (tokenizer, model, device, label_list, model_info)
        model_info = {"type": "standard"|"crf"|"dual_head", "is_bioes": bool}
    """
    model_dir = str(model_dir)

    # ── Device ────────────────────────────────────────────────────
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # ── Config okuma ──────────────────────────────────────────────
    config_path = Path(model_dir) / "config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # ── Label list (dinamik, config.json'dan) ─────────────────────
    id2label = cfg.get("id2label", {"0": "O", "1": "B-TERM", "2": "I-TERM"})
    label_list = [id2label[str(i)] for i in range(len(id2label))]
    is_bioes = "E-TERM" in label_list or "S-TERM" in label_list
    is_crf = cfg.get("crf", False)
    is_dual_head = cfg.get("dual_head", False)

    # ── Model yükleme ────────────────────────────────────────────
    if is_crf:
        from bert_crf_model import BertCRFForNER
        model = BertCRFForNER.from_pretrained(model_dir)
        model_type = "crf"
        logger.info("BERT-CRF model loaded from %s", model_dir)
    elif is_dual_head:
        from dual_head_model import BertForDualHeadNER
        model = BertForDualHeadNER.from_pretrained(model_dir)
        model_type = "dual_head"
        logger.info("Dual-head model loaded from %s", model_dir)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        model_type = "standard"

    model.to(device)
    model.eval()

    model_info = {"type": model_type, "is_bioes": is_bioes}
    scheme = "CRF" if is_crf else ("BIOES" if is_bioes else "BIO")
    logger.info("Model loaded: %s, scheme=%s (%d labels), device=%s",
                model_type, scheme, len(label_list), device)

    return tokenizer, model, device, label_list, model_info


def extract_preds(model, inputs, model_info):
    """Model çıktısından tahmin ID listesi çıkar (tüm model tipleri).

    CRF → Viterbi decode, dual-head → bio_logits argmax, standard → logits argmax.
    [CLS] ve [SEP] dahil tüm pozisyonlar için tahmin döner.

    Args:
        model: Yüklenmiş model.
        inputs: Tokenizer çıktısı (device'a taşınmış, offset_mapping pop edilmiş).
        model_info: load_ner_model()'dan dönen model_info dict.

    Returns:
        (preds, logits_or_none)
        preds: list[int] — her pozisyon için tahmin edilen label ID.
        logits_or_none: Tensor veya None — softmax için (CRF'de None).
    """
    # Guard: empty/short input (only [CLS]+[SEP] or less)
    attn = inputs["attention_mask"][0]
    seq_len = int(attn.sum().item())
    seq_total = attn.size(0)
    if seq_len <= 2:
        return [0] * seq_total, None

    with torch.no_grad():
        outputs = model(**inputs)

    if model_info["type"] == "crf":
        emissions = outputs["logits"]
        # [CLS] (pos 0) kes — torchcrf ilk pozisyon mask=True gerektirir
        emissions_nocls = emissions[:, 1:, :]
        # Mask: attention_mask[1:] — [SEP] dahil ama [CLS] hariç
        crf_mask = attn[1:].bool().clone()
        # [SEP]'i de maskele (son gerçek pozisyon)
        crf_mask[seq_len - 2] = False  # seq_len-1 orijinalde [SEP], -1 shift sonrası seq_len-2

        decoded = model.decode(emissions_nocls, mask=crf_mask.unsqueeze(0))[0]

        # Map back: pos 0 = [CLS] → O, pos 1..N = decoded + padding
        preds = [0] * emissions.size(1)  # all O
        mask_positions = crf_mask.nonzero(as_tuple=True)[0].tolist()
        for j, mpos in enumerate(mask_positions):
            if j < len(decoded):
                preds[mpos + 1] = decoded[j]  # +1 for [CLS] offset
        return preds, None

    elif model_info["type"] == "dual_head":
        logits = outputs["bio_logits"]
        preds = torch.argmax(logits, dim=-1)[0].tolist()
        return preds, logits[0]

    else:
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)[0].tolist()
        return preds, logits[0]


def decode_entities(preds, offsets, label_list, input_text, prefix_len,
                    logits_for_scores=None):
    """Tahmin ID dizisinden karakter-düzeyi entity listesi çıkar.

    BIO ve BIOES şemalarını destekler:
      - B-TERM: yeni entity başlat
      - I-TERM: mevcut entity'yi uzat
      - S-TERM: tek-tokenli entity (BIOES)
      - E-TERM: entity sonu (BIOES) — mevcut entity'yi kapat
      - O / diğer: entity kapat

    Args:
        logits_for_scores: Softmax probability tensor for confidence scores.
            None ise (CRF modelleri) tüm entity score'ları 1.0 olur.
            CRF Viterbi decode per-token probability üretmez.

    Returns:
        list of {"start": int, "end": int, "text": str, "score": float}
        Offset'ler prefix_len kadar düzeltilmiş (negatif olanlar filtrelenmiş).
    """
    entities = []
    current_entity = None
    current_scores = []

    for i, (pred_id, (start, end)) in enumerate(zip(preds, offsets)):
        if start == 0 and end == 0:
            # Special token ([CLS], [SEP], padding)
            if current_entity:
                current_entity["score"] = sum(current_scores) / len(current_scores) if current_scores else 0.0
                entities.append(current_entity)
                current_entity = None
                current_scores = []
            continue

        label = label_list[pred_id]
        score = logits_for_scores[i][pred_id].item() if logits_for_scores is not None else 1.0

        if label == "S-TERM":
            # Tek-tokenli entity (BIOES)
            if current_entity:
                current_entity["score"] = sum(current_scores) / len(current_scores) if current_scores else 0.0
                entities.append(current_entity)
            entities.append({
                "start": start, "end": end,
                "text": input_text[start:end], "score": score,
            })
            current_entity = None
            current_scores = []

        elif label == "B-TERM":
            if current_entity:
                current_entity["score"] = sum(current_scores) / len(current_scores) if current_scores else 0.0
                entities.append(current_entity)
            current_entity = {"start": start, "end": end, "text": input_text[start:end]}
            current_scores = [score]

        elif label == "I-TERM" and current_entity:
            current_entity["end"] = end
            current_entity["text"] = input_text[current_entity["start"]:end]
            current_scores.append(score)

        elif label == "E-TERM" and current_entity:
            # Entity sonu (BIOES)
            current_entity["end"] = end
            current_entity["text"] = input_text[current_entity["start"]:end]
            current_scores.append(score)
            current_entity["score"] = sum(current_scores) / len(current_scores)
            entities.append(current_entity)
            current_entity = None
            current_scores = []

        else:
            # O veya orphan I/E
            if current_entity:
                current_entity["score"] = sum(current_scores) / len(current_scores) if current_scores else 0.0
                entities.append(current_entity)
                current_entity = None
                current_scores = []

    if current_entity:
        current_entity["score"] = sum(current_scores) / len(current_scores) if current_scores else 0.0
        entities.append(current_entity)

    # Prefix offset düzeltmesi
    for ent in entities:
        ent["start"] -= prefix_len
        ent["end"] -= prefix_len

    # Negatif offset'li entity'leri filtrele (prefix içine düşenler)
    return [e for e in entities if e["start"] >= 0]
