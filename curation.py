"""CuratorKIT pieces for building the 5-form parallel safety set.

* ``OpenMTTranslator``: English -> Devanagari Hindi with an open ~1B MT model
  (IndicTrans2-1B by default, NLLB-200-distilled-1.3B as the alternative).
* ``export_llm_batches`` / ``load_llm_variants``: the hand-off to an external LLM,
  which writes the paraphrase, Roman Hindi and Hinglish forms (see
  ``LLM_DATA_PROMPT.md``).
* ``TranslatorRefusalGate`` / ``ScriptGate`` / ``FidelityGate``: CuratorKIT gates
  that reject variants that were refused, written in the wrong script, copied, or
  drifted in meaning. Every rejection carries a reason string.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from curatorkit.interfaces import BaseGate
from curatorkit.schema import DataSample, RejectedSample

from mlsafety import english_overlap, is_refusal_heuristic, script_stats, word_jaccard

VARIANT_FORMS = ["en_para", "hi_deva", "hi_roman", "hinglish"]
LLM_FORMS = ["en_para", "hi_roman", "hinglish"]

MT_MODELS = {
    "indictrans2": "ai4bharat/indictrans2-en-indic-1B",
    "nllb": "facebook/nllb-200-distilled-1.3B",
}


# --------------------------------------------------------------------------- #
# Open machine translation (English -> hi_Deva)                               #
# --------------------------------------------------------------------------- #

class OpenMTTranslator:
    """Batched English -> Hindi (Devanagari) with IndicTrans2 or NLLB."""

    def __init__(self, backend: str = "indictrans2", model_id: Optional[str] = None,
                 num_beams: int = 5, max_length: int = 256, device: Optional[str] = None) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if backend not in MT_MODELS:
            raise ValueError(f"backend must be one of {list(MT_MODELS)}")
        self.backend = backend
        self.model_id = model_id or MT_MODELS[backend]
        self.num_beams, self.max_length = num_beams, max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        remote = backend == "indictrans2"
        self.tok = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=remote, **({} if remote else {"src_lang": "eng_Latn"})
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, trust_remote_code=remote, torch_dtype=dtype
        ).to(self.device).eval()
        if remote:
            from IndicTransToolkit import IndicProcessor

            self.ip = IndicProcessor(inference=True)

    def _translate_batch(self, texts: list[str]) -> list[str]:
        import torch

        if self.backend == "indictrans2":
            texts = self.ip.preprocess_batch(texts, src_lang="eng_Latn", tgt_lang="hin_Deva")
        enc = self.tok(texts, truncation=True, padding="longest", return_tensors="pt",
                       return_attention_mask=True).to(self.device)
        kw: dict[str, Any] = {"num_beams": self.num_beams, "max_length": self.max_length, "use_cache": True}
        if self.backend == "nllb":
            kw["forced_bos_token_id"] = self.tok.convert_tokens_to_ids("hin_Deva")
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        dec = self.tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if self.backend == "indictrans2":
            dec = self.ip.postprocess_batch(dec, lang="hin_Deva")
        return [d.strip() for d in dec]

    def translate(self, texts: Sequence[str], batch_size: int = 16) -> list[str]:
        from tqdm.auto import tqdm

        out: list[str] = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"{self.backend} en->hi"):
            out += self._translate_batch(list(texts[i:i + batch_size]))
        return out


# --------------------------------------------------------------------------- #
# Hand-off to the external LLM                                                #
# --------------------------------------------------------------------------- #

def export_llm_batches(items: list[dict], out_dir: str | Path, batch_size: int = 25) -> list[Path]:
    """Write ``[{id, en, hi_deva}, ...]`` as numbered JSON batches to paste into the LLM."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("batch_*.json"):
        old.unlink()
    paths = []
    for b, i in enumerate(range(0, len(items), batch_size)):
        p = out_dir / f"batch_{b:03d}.json"
        p.write_text(json.dumps(items[i:i + batch_size], ensure_ascii=False, indent=1), encoding="utf-8")
        paths.append(p)
    return paths


def _parse_llm_file(text: str) -> list[dict]:
    text = re.sub(r"```(?:json|jsonl)?", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        pass
    rows = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_llm_variants(response_dir: str | Path) -> dict[str, dict]:
    """Read every ``*.json`` / ``*.jsonl`` / ``*.txt`` LLM response in a folder; later files win."""
    found: dict[str, dict] = {}
    for p in sorted(Path(response_dir).glob("*")):
        if p.suffix.lower() not in {".json", ".jsonl", ".txt"}:
            continue
        for r in _parse_llm_file(p.read_text(encoding="utf-8")):
            if isinstance(r, dict) and "id" in r:
                found[str(r["id"])] = {**r, "_file": p.name}
    return found


# --------------------------------------------------------------------------- #
# Gates                                                                       #
# --------------------------------------------------------------------------- #

def _reject(sample: DataSample, reason: str, step: str) -> RejectedSample:
    return RejectedSample(**sample.model_dump(), rejection_reason=reason, rejecting_step=step)


class TranslatorRefusalGate(BaseGate):
    """Drop variants that are refusals, or answers to the request instead of rewrites."""

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed, rejected = [], []
        for s in samples:
            forms = s.metadata["forms"]
            bad = [k for k in VARIANT_FORMS if is_refusal_heuristic(forms[k])]
            too_long = [k for k in VARIANT_FORMS if len(forms[k].split()) > 3 * len(forms["en"].split()) + 15]
            if bad:
                rejected.append(_reject(s, f"variant_refused:{','.join(bad)}", type(self).__name__))
            elif too_long:
                rejected.append(_reject(s, f"variant_answered:{','.join(too_long)}", type(self).__name__))
            else:
                passed.append(s)
        return passed, rejected


class ScriptGate(BaseGate):
    """Each form must be in the script and register it claims to be."""

    def __init__(
        self,
        min_devanagari: float = 0.6,
        max_devanagari_latin: float = 0.02,
        min_hindi_words_roman: int = 2,
        min_hindi_words_hinglish: int = 1,
        min_english_kept_hinglish: int = 1,
        max_copy_paraphrase: float = 0.7,
        max_copy_hinglish_roman: float = 0.9,
    ) -> None:
        self.min_devanagari = min_devanagari
        self.max_devanagari_latin = max_devanagari_latin
        self.min_hindi_words_roman = min_hindi_words_roman
        self.min_hindi_words_hinglish = min_hindi_words_hinglish
        self.min_english_kept_hinglish = min_english_kept_hinglish
        self.max_copy_paraphrase = max_copy_paraphrase
        self.max_copy_hinglish_roman = max_copy_hinglish_roman

    def check(self, forms: dict[str, str]) -> list[str]:
        problems = []
        deva = script_stats(forms["hi_deva"])
        if deva["devanagari_frac"] < self.min_devanagari:
            problems.append(f"hi_deva_not_devanagari:{deva['devanagari_frac']:.2f}")
        for key in ("en_para", "hi_roman", "hinglish"):
            st = script_stats(forms[key])
            if st["devanagari_frac"] > self.max_devanagari_latin:
                problems.append(f"{key}_has_devanagari:{st['devanagari_frac']:.2f}")
        for key, min_fw in (("hi_roman", self.min_hindi_words_roman), ("hinglish", self.min_hindi_words_hinglish)):
            n = script_stats(forms[key])["hindi_function_words"]
            if n < min_fw:
                problems.append(f"{key}_not_hindi:{n}")
        if english_overlap(forms["hinglish"], forms["en"]) < self.min_english_kept_hinglish:
            problems.append("hinglish_no_english_terms")
        copy = word_jaccard(forms["en_para"], forms["en"])
        if copy > self.max_copy_paraphrase:
            problems.append(f"en_para_copies_seed:{copy:.2f}")
        same = word_jaccard(forms["hinglish"], forms["hi_roman"])
        if same > self.max_copy_hinglish_roman:
            problems.append(f"hinglish_same_as_hi_roman:{same:.2f}")
        return problems

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed, rejected = [], []
        for s in samples:
            problems = self.check(s.metadata["forms"])
            if problems:
                rejected.append(_reject(s, "script:" + ";".join(problems), type(self).__name__))
            else:
                passed.append(s)
        return passed, rejected


class FidelityGate(BaseGate):
    """Multilingual-embedding similarity between each variant and the English seed.

    Only forms listed in ``thresholds`` are gated. LaBSE was not trained on Romanized
    Hindi, so its similarity there tracks how many English words survive more than
    meaning; those forms are scored for the dataset card (``metadata["fidelity"]``)
    but left to ``ScriptGate`` and the human review.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        thresholds: Optional[dict[str, float]] = None,
        device: Optional[str] = None,
        encoder: Any = None,
    ) -> None:
        self.model_name = model_name
        self.thresholds = thresholds or {"en_para": 0.75, "hi_deva": 0.70}
        self.device = device
        self._encoder = encoder

    def _encode(self, texts: list[str]):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name, device=self.device)
        return self._encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        if not samples:
            return [], []
        keys = ["en"] + VARIANT_FORMS
        flat = [s.metadata["forms"][k] for s in samples for k in keys]
        emb = self._encode(flat).reshape(len(samples), len(keys), -1)
        passed, rejected = [], []
        for s, e in zip(samples, emb):
            sims = {k: float(e[0] @ e[i + 1]) for i, k in enumerate(VARIANT_FORMS)}
            s.metadata["fidelity"] = sims
            low = [f"{k}:{v:.2f}" for k, v in sims.items() if k in self.thresholds and v < self.thresholds[k]]
            if low:
                rejected.append(_reject(s, "low_fidelity:" + ",".join(low), type(self).__name__))
            else:
                passed.append(s)
        return passed, rejected


# --------------------------------------------------------------------------- #
# Conversions                                                                 #
# --------------------------------------------------------------------------- #

def seeds_to_samples(seeds: list[dict], source_uri: str = "seeds") -> list[DataSample]:
    """Seed rows ``{family_id, kind, source, prompt, ...}`` -> CuratorKIT ``DataSample``s."""
    return [
        DataSample(
            source_uri=r.get("source_uri", source_uri),
            instruction=r["prompt"],
            task_type="prompt_only",
            metadata={k: v for k, v in r.items() if k != "prompt"},
        )
        for r in seeds
    ]


def seed_key(seed: dict) -> str:
    return f"{seed['family_id']}-{seed['kind']}"


def rejected_summary(rejected: list[RejectedSample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rejected:
        key = r.rejection_reason.split(":")[0]
        out[key] = out.get(key, 0) + 1
    return out
