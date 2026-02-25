#!/usr/bin/env python3
"""Local test for ml_backend — works without Flask.

Bypasses the Flask/werkzeug ast.Str issue on Python 3.14.
Tests model.py inference directly.

Usage:
    cd ml_backend
    MODEL_DIR=./model python test_local.py
"""

import os
import sys
import json
import time

os.environ.setdefault("MODEL_DIR", "./model")

from model import load_model, predict_tasks  # noqa: E402

LABEL_CONFIG = (
    '<View>'
    '<Labels name="label" toName="text">'
    '<Label value="TERM" background="red"/>'
    '</Labels>'
    '<Text name="text" value="$text" granularity="word"/>'
    '</View>'
)


def test_short():
    """Short text — single window."""
    tasks = [
        {
            "data": {
                "text": "İvme hızın zamana göre değişim oranıdır.",
                "branch": "Fizik",
            }
        }
    ]
    results = predict_tasks(tasks, LABEL_CONFIG)
    terms = [r["value"]["text"] for r in results[0]["result"]]
    print(f"  Terms: {terms}")
    assert results[0]["model_version"] == "v001"
    assert results[0]["result"][0]["from_name"] == "label"
    assert results[0]["result"][0]["to_name"] == "text"
    assert results[0]["result"][0]["type"] == "labels"


def test_long():
    """Long text — sentence chunking."""
    sentences = [
        "İvme hızın zamana göre değişim oranıdır.",
        "Kuvvet kütle ile ivmenin çarpımına eşittir.",
        "Enerji bir sistemin iş yapabilme kapasitesidir.",
        "Kinetik enerji hareket enerjisi, potansiyel enerji ise konum enerjisidir.",
        "Elektromanyetik dalgalar elektrik ve manyetik alanların birbirini oluşturmasıyla yayılır.",
        "Frekans birim zamandaki titreşim sayısıdır.",
        "Dalga boyu ardışık iki tepe noktası arasındaki uzaklıktır.",
        "Fotosentez bitkilerin ışık enerjisini kimyasal enerjiye dönüştürdüğü süreçtir.",
        "Hücre canlının en küçük yapı ve işlev birimidir.",
        "Osmoz suyun yarı geçirgen zardan geçişidir.",
    ]
    long_text = " ".join(sentences * 6)
    tasks = [{"data": {"text": long_text, "branch": "Fizik"}}]

    t0 = time.time()
    results = predict_tasks(tasks, LABEL_CONFIG)
    elapsed = time.time() - t0

    ents = results[0]["result"]
    print(f"  Entities: {len(ents)}, time: {elapsed:.2f}s")
    if ents:
        last_end = ents[-1]["value"]["end"]
        pct = 100 * last_end / len(long_text)
        print(f"  Coverage: {last_end}/{len(long_text)} ({pct:.0f}%)")
    assert len(ents) > 50, f"Expected >50 entities, got {len(ents)}"


def test_oversized_sentence():
    """Single very long sentence with no period — must not truncate."""
    # ~500 tokens, no sentence-ending punctuation
    words = (
        "ivme hız kuvvet kütle enerji potansiyel kinetik elektromanyetik "
        "dalga frekans atom molekül hücre osmoz fotosentez klorofil "
        "mitokondri ribozom çekirdek sitoplazma zarı geçirgen difüzyon "
    )
    long_sentence = (words * 25).strip()  # ~600 tokens, one "sentence"
    tasks = [{"data": {"text": long_sentence, "branch": "Fizik"}}]

    results = predict_tasks(tasks, LABEL_CONFIG)
    ents = results[0]["result"]
    print(f"  Entities: {len(ents)}")
    if ents:
        last_end = ents[-1]["value"]["end"]
        pct = 100 * last_end / len(long_sentence)
        print(f"  Last entity at {last_end}/{len(long_sentence)} ({pct:.0f}%)")
        assert pct > 50, f"Coverage {pct:.0f}% too low — truncation detected"


def test_no_branch():
    """Text without branch field."""
    tasks = [
        {
            "data": {
                "text": "Fotosentez bitkilerin ışık enerjisini kimyasal enerjiye dönüştürdüğü süreçtir."
            }
        }
    ]
    results = predict_tasks(tasks, LABEL_CONFIG)
    terms = [r["value"]["text"] for r in results[0]["result"]]
    print(f"  Terms: {terms}")


def test_label_config_fallback():
    """No label_config → should use defaults with warning."""
    tasks = [{"data": {"text": "Atom yapısı."}}]
    results = predict_tasks(tasks, None)
    print(f"  Result count: {len(results[0]['result'])}")


def test_label_studio_format():
    """Verify the response matches Label Studio's expected format."""
    tasks = [{"data": {"text": "Kuvvet kütle ile ivmenin çarpımına eşittir.", "branch": "Fizik"}}]
    results = predict_tasks(tasks, LABEL_CONFIG)
    pred = results[0]

    # Top-level keys
    assert set(pred.keys()) == {"result", "score", "model_version"}
    assert isinstance(pred["score"], float)
    assert isinstance(pred["model_version"], str)

    if pred["result"]:
        annot = pred["result"][0]
        assert set(annot.keys()) == {"from_name", "to_name", "type", "value", "score"}
        assert annot["type"] == "labels"
        val = annot["value"]
        assert set(val.keys()) == {"start", "end", "text", "labels"}
        assert val["labels"] == ["TERM"]
        assert isinstance(val["start"], int)
        assert isinstance(val["end"], int)
    print("  Format OK")


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    print()

    tests = [
        ("Short text", test_short),
        ("Long text (sentence chunking)", test_long),
        ("Oversized single sentence", test_oversized_sentence),
        ("No branch", test_no_branch),
        ("Label config fallback", test_label_config_fallback),
        ("Label Studio format", test_label_studio_format),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"[TEST] {name}")
        try:
            fn()
            passed += 1
            print(f"  PASS\n")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}\n")

    print(f"{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
