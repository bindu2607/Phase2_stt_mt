# mt_for_tts_full.py
from __future__ import annotations
"""
MT -> TTS full pipeline (single-file)
- Bi-directional EN <-> ZH interactive demo (choose source and target language)
- Idiom/phrase handler to expand common idioms before MT
- Outputs per-chunk JSON + TTS-ready txt + SSML into mt_outputs/
"""
import re
import time
import math
import copy
import json
import os
import pathlib
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Optional heavy imports - handled defensively
try:
    import numpy as np
except Exception:
    raise RuntimeError("numpy is required")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

# optional punctuation model
try:
    from deepmultilingualpunctuation import PunctuationModel
    _HAS_PUNCT = True
except Exception:
    PunctuationModel = None
    _HAS_PUNCT = False

# optional cachetools
try:
    from cachetools import LRUCache
    _HAS_CACHETOOLS = True
except Exception:
    LRUCache = dict
    _HAS_CACHETOOLS = False

# logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("MTForTTSFull")

# output dir
OUT_DIR = pathlib.Path("mt_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Data classes
# -------------------------
@dataclass
class Token:
    text: str
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    asr_conf: Optional[float] = None
    orig_index: Optional[int] = None
    is_overlap: bool = False

@dataclass
class ChunkMessage:
    session_id: str
    chunk_id: int
    is_final: bool
    src_lang: str
    tgt_lang: str
    chunk_start_ms: int
    chunk_end_ms: int
    tokens: List[Token]
    raw_text: str

@dataclass
class TTSChunk:
    session_id: str
    chunk_id: int
    translated_text: str
    tts_text: str
    ssml: Optional[str]
    source_words: List[str]
    target_words: List[str]
    word_mapping: List[List[int]]
    target_word_timestamps: Optional[List[Tuple[int,int]]]
    pause_hints: List[Dict[str,int]]
    token_confidences: List[float]
    confidence: float
    model_id: str
    cache_hit: bool
    prosody: Dict[str, Any]
    metadata: Dict[str, Any]
    replace_chunk_id: Optional[int] = None
    partial: bool = False

# -------------------------
# Utilities
# -------------------------
def now_s() -> float:
    return time.time()

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def clamp(v, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo

def softmax_numpy(logits):
    a = np.asarray(logits, dtype=float)
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()

def write_outputs_to_disk(tts_chunk: TTSChunk):
    """Save JSON, TTS txt and SSML for downstream TTS ingestion."""
    ts = now_ts()
    base = OUT_DIR / f"{ts}_sess{tts_chunk.session_id}_chunk{tts_chunk.chunk_id}"
    # json
    try:
        j = asdict(tts_chunk)
        with open(str(base) + ".json", "w", encoding="utf-8") as f:
            json.dump(j, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("Failed saving JSON output.")
    # tts txt
    try:
        with open(str(base) + ".txt", "w", encoding="utf-8") as f:
            f.write(tts_chunk.tts_text or tts_chunk.translated_text or "")
    except Exception:
        logger.exception("Failed saving TTS txt.")
    # ssml
    try:
        if tts_chunk.ssml:
            with open(str(base) + ".ssml", "w", encoding="utf-8") as f:
                f.write(tts_chunk.ssml)
    except Exception:
        logger.exception("Failed saving SSML.")

# -------------------------
# Preprocessor
# -------------------------
DEFAULT_FILLERS = {"um","uh","hmm","mm","erm","ah","uhm","you know","i mean","like","sort of","kind of"}
ARTIFACT_RE = re.compile(r"(\[.*?\]|<.*?>|\d{1,2}:\d{2}(:\d{2})?)")

class Preprocessor:
    def __init__(self, fillers: Optional[set]=None, min_token_len:int=1):
        self.fillers = fillers or DEFAULT_FILLERS
        self.min_token_len = min_token_len

    def preprocess_chunk(self, chunk: ChunkMessage):
        text = (chunk.raw_text or "")
        text = re.sub(ARTIFACT_RE, "", text)
        text = re.sub(r"\s{2,}", " ", text).strip()

        pp_tokens = []
        if chunk.tokens:
            for i,t in enumerate(chunk.tokens):
                txt = t.text.strip()
                if not txt:
                    continue
                if txt.lower() in self.fillers or len(txt) < self.min_token_len:
                    continue
                tok = Token(text=txt, start_ms=t.start_ms, end_ms=t.end_ms, asr_conf=t.asr_conf, orig_index=i)
                pp_tokens.append(tok)
        else:
            for i,w in enumerate(text.split()):
                if not w or w.lower() in self.fillers or len(w) < self.min_token_len:
                    continue
                pp_tokens.append(Token(text=w, orig_index=i))
        # interpolate timestamps if missing
        any_ts = any((p.start_ms is not None and p.end_ms is not None) for p in pp_tokens)
        if not any_ts and pp_tokens:
            total_ms = max(1, (chunk.chunk_end_ms - chunk.chunk_start_ms) or 1)
            per = total_ms / len(pp_tokens)
            for i,p in enumerate(pp_tokens):
                p.start_ms = int(chunk.chunk_start_ms + i * per)
                p.end_ms = int(chunk.chunk_start_ms + (i+1) * per)
        pp_text = " ".join([p.text for p in pp_tokens])
        return pp_tokens, pp_text

# -------------------------
# Chunk Manager
# -------------------------
class ChunkManager:
    def __init__(self, pause_threshold_ms:int=600, overlap_ms:int=250):
        self.pause_threshold_ms = pause_threshold_ms
        self.overlap_ms = overlap_ms
        self.finalized_chunks: List[ChunkMessage] = []

    def attach_overlap(self, chunk: ChunkMessage) -> ChunkMessage:
        if not self.finalized_chunks:
            return chunk
        prev = self.finalized_chunks[-1]
        ov_tokens = []
        for t in reversed(prev.tokens):
            if t.end_ms is None:
                break
            if prev.chunk_end_ms - t.end_ms <= self.overlap_ms:
                tt = copy.deepcopy(t)
                tt.is_overlap = True
                ov_tokens.insert(0, tt)
            else:
                break
        if ov_tokens:
            for t in ov_tokens:
                t.orig_index = None
            chunk.tokens = ov_tokens + chunk.tokens
        return chunk

    def finalize_and_record(self, chunk: ChunkMessage):
        self.finalized_chunks.append(chunk)
        if len(self.finalized_chunks) > 50:
            self.finalized_chunks.pop(0)

    def get_context_text(self, n_chunks=2) -> str:
        parts = [c.raw_text for c in self.finalized_chunks[-n_chunks:]]
        return " ".join(parts).strip()

# -------------------------
# Idiom / phrase handler (updated for EN↔ZH)
# -------------------------
class IdiomHandler:
    """Expand common idioms/phrases to paraphrases (helps MT produce natural TTS).
       Supports English->English paraphrase, English->Chinese equivalents, Chinese->Chinese,
       and Chinese->English equivalents so EN<->ZH flows benefit.
    """

    def __init__(self):
        # English idiom -> English paraphrase
        self.eng_map = {
            r"\bbreak the ice\b": "make people feel more comfortable",
            r"\bpiece of cake\b": "very easy",
            r"\bover the moon\b": "extremely happy",
            r"\bbite the bullet\b": "accept something difficult",
            r"\bspill the beans\b": "reveal the secret",
        }
        # English idiom -> common Chinese equivalent (useful when translating EN -> ZH)
        self.eng_to_zh_map = {
            r"\bbreak the ice\b": "打破僵局",
            r"\bpiece of cake\b": "小菜一碟",
            r"\bover the moon\b": "欣喜若狂",
            r"\bbite the bullet\b": "咬紧牙关",
            r"\bspill the beans\b": "露馅",
        }
        # Chinese idiom -> Chinese paraphrase (simplified)
        self.zhm_map = {
            "画蛇添足": "做了多余的事",
            "守株待兔": "不主动努力，盲等机遇",
            "杯弓蛇影": "疑神疑鬼",
            "一箭双雕": "同时达到两个目的",
        }
        # Chinese idiom -> English equivalent/paraphrase (useful when translating ZH -> EN)
        self.zhm_to_eng_map = {
            "画蛇添足": "do something unnecessary",
            "守株待兔": "wait idly for opportunities",
            "杯弓蛇影": "be suspicious without cause",
            "一箭双雕": "kill two birds with one stone",
        }

    def expand(self, text: str, src_lang_code: str, tgt_lang_code: Optional[str] = None) -> str:
        """
        Expand idioms based on source and (optionally) target language.
        Behavior:
          - EN source:
            - If tgt is ZH: replace idiom by English paraphrase + (Chinese equivalent) to help MT.
            - Otherwise: replace idiom by English paraphrase.
          - ZH source:
            - If tgt is EN: replace Chinese idiom by Chinese paraphrase + (English equivalent).
            - Otherwise: replace Chinese idiom by Chinese paraphrase.
        """
        t = text

        # Normalize codes to simple forms for checks
        s_src = (src_lang_code or "").lower()
        s_tgt = (tgt_lang_code or "").lower() if tgt_lang_code else ""

        # English source handling
        if s_src.startswith("eng") or s_src.startswith("en"):
            # EN -> ZH: append Chinese equivalent in parentheses (if exists)
            for pattern, para in self.eng_map.items():
                if re.search(pattern, t, flags=re.IGNORECASE):
                    zh_equiv = None
                    # check zho/zh too
                    if (s_tgt.startswith("zho") or s_tgt.startswith("zh")):
                        zh_equiv = self.eng_to_zh_map.get(pattern)
                    if zh_equiv:
                        t = re.sub(pattern, f"{para}（{zh_equiv}）", t, flags=re.IGNORECASE)
                    else:
                        t = re.sub(pattern, para, t, flags=re.IGNORECASE)
            return t

        # Chinese source handling
        if s_src.startswith("zho") or s_src.startswith("zh"):
            # If target is English and English equiv exists, append english eq in parentheses
            for zh, para in self.zhm_map.items():
                if zh in t:
                    eng_equiv = self.zhm_to_eng_map.get(zh)
                    if s_tgt.startswith("eng") or s_tgt.startswith("en"):
                        if eng_equiv:
                            t = t.replace(zh, f"{para} ({eng_equiv})")
                        else:
                            t = t.replace(zh, para)
                    else:
                        t = t.replace(zh, para)
            return t

        # fallback: no change
        return t

# -------------------------
# Punctuator / Truecaser / Normalizer
# -------------------------
class PunctuatorTruecaserNormalizer:
    def __init__(self, spell_out_numbers:bool=False):
        self.spell_out_numbers = spell_out_numbers
        self._punct_model = None
        if _HAS_PUNCT:
            try:
                self._punct_model = PunctuationModel()
                logger.info("Loaded deepmultilingualpunctuation model.")
            except Exception as e:
                logger.warning("Failed loading punctuator: %s", e)
                self._punct_model = None

    @staticmethod
    def _heuristic_punctuate(text: str, is_final: bool) -> str:
        t = text.strip()
        if not t: return t
        t = t[0].upper() + t[1:]
        if t[-1] not in ".!?":
            if is_final:
                t = t + "."
        return t

    @staticmethod
    def _insert_commas_on_pauses(tokens: List[Token], comma_threshold_ms:int=250) -> str:
        out = []
        for i,t in enumerate(tokens):
            out.append(t.text)
            if i+1 < len(tokens):
                gap = 0
                if tokens[i+1].start_ms is not None and t.end_ms is not None:
                    gap = tokens[i+1].start_ms - t.end_ms
                if gap and gap >= comma_threshold_ms:
                    out.append(",")
        return " ".join(out)

    def normalize_numbers(self, text:str) -> str:
        if not self.spell_out_numbers:
            return text
        try:
            import num2words
        except Exception:
            return text
        def repl(m):
            try:
                return num2words.num2words(int(m.group(0)))
            except Exception:
                return m.group(0)
        return re.sub(r"\b\d+\b", repl, text)

    def punctuate_truecase_normalize(self, chunk: ChunkMessage, pp_tokens: List[Token], is_final: bool, comma_threshold_ms:int=250):
        raw = " ".join([t.text for t in pp_tokens])
        punct_text = None
        if self._punct_model:
            try:
                punct_text = self._punct_model.restore_punctuation(raw)
            except Exception as e:
                logger.debug("Punctuator error: %s", e)
                punct_text = None
        if punct_text is None:
            punct_text = self._insert_commas_on_pauses(pp_tokens, comma_threshold_ms=comma_threshold_ms)
            punct_text = self._heuristic_punctuate(punct_text, is_final=is_final)

        words = punct_text.split()
        orig_idx_by_word = []
        i_tok = 0
        for w in words:
            w_clean = re.sub(r"^[,.:;!?]+|[,.:;!?]+$", "", w)
            matched_idx = None
            # substring heuristic (robust)
            while i_tok < len(pp_tokens):
                candidate = pp_tokens[i_tok].text
                if candidate and candidate.lower().startswith(w_clean[:min(3,len(w_clean))].lower()):
                    matched_idx = pp_tokens[i_tok].orig_index
                    i_tok += 1
                    break
                else:
                    i_tok += 1
            orig_idx_by_word.append(matched_idx if matched_idx is not None else (0 if pp_tokens else None))

        if words:
            if len(words[0]) > 0:
                words[0] = words[0][0].upper() + words[0][1:] if len(words[0])>1 else words[0].upper()
        punct_text = " ".join(words)
        punct_text = self.normalize_numbers(punct_text)
        return punct_text, orig_idx_by_word

# -------------------------
# Context Manager + Cache
# -------------------------
class ContextManager:
    def __init__(self, context_chunks:int=2, cache_maxsize:int=4096):
        self.context_chunks = context_chunks
        if _HAS_CACHETOOLS:
            self.cache = LRUCache(maxsize=cache_maxsize)
        else:
            self.cache = {}
        self.finalized_chunks_texts: List[str] = []

    def push_finalized_chunk(self, chunk:ChunkMessage):
        self.finalized_chunks_texts.append(chunk.raw_text)
        if len(self.finalized_chunks_texts) > 50:
            self.finalized_chunks_texts.pop(0)

    def get_context(self) -> str:
        return " ".join(self.finalized_chunks_texts[-self.context_chunks:]).strip()

    def make_cache_key(self, context_text:str, punct_text:str, src_lang:str, tgt_lang:str) -> str:
        norm = re.sub(r"\s+", " ", f"{context_text} {punct_text}".strip()).lower()
        return f"{src_lang}|{tgt_lang}|{norm}"

    def cache_get(self, key:str):
        return self.cache.get(key, None)

    def cache_set(self, key:str, value:Any):
        try:
            self.cache[key] = value
        except Exception:
            self.cache[key] = value

# -------------------------
# MT service wrapper (NLLB)
# -------------------------
class MTService:
    def __init__(self, model_name_preference:List[str]=None, device:Optional[str]=None, max_workers:int=2):
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise RuntimeError("transformers and torch are required for MTService")
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        preferred = model_name_preference or ["facebook/nllb-200-1.3B", "facebook/nllb-200-distilled-600M"]
        for name in preferred:
            try:
                logger.info("Attempting to load model %s on %s ...", name, self.device)
                tok = AutoTokenizer.from_pretrained(name, use_fast=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(name)
                try:
                    model.to(self.device)
                except Exception:
                    logger.debug("Could not move model %s to device %s", name, self.device)
                self.model = model
                self.tokenizer = tok
                self.model_name = name
                logger.info("Loaded MT model %s", name)
                break
            except Exception as e:
                logger.warning("Failed loading model %s: %s", name, str(e)[:200])
                continue
        if self.model is None:
            raise RuntimeError("Failed to load any MT model from preferences: " + ", ".join(preferred))
        self.default_gen_kwargs = {
            "num_beams": 5,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "max_length": 256,
            "return_dict_in_generate": True,
            "output_attentions": True,
            "output_scores": True,
            "remove_invalid_values": True
        }
        self.lang_code_map = getattr(self.tokenizer, "lang_code_to_id", None)

    def _prepare_inputs(self, text:str):
        return self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    def translate_sync(self, src_text:str, src_lang:str, tgt_lang:str, context:Optional[str]=None, gen_kwargs:Optional[dict]=None):
        """
        Robust translate_sync that tries hard to set forced_bos_token_id for target language.
        Expects src_lang and tgt_lang may be canonical codes or short forms; will log mapping keys for debugging.
        """
        kwargs = dict(self.default_gen_kwargs)
        if gen_kwargs:
            kwargs.update(gen_kwargs)
        input_text = f"{context} {src_text}".strip() if context else src_text

        # Debug log: what we're sending to MT
        logger.debug("MTService.translate_sync input_text (first200): %s", input_text[:200])
        logger.debug("MTService.translate_sync src_lang=%s tgt_lang=%s", src_lang, tgt_lang)
        logger.debug("MTService.translate_sync lang_code_map keys sample=%s", 
                     list(self.lang_code_map.keys())[:20] if self.lang_code_map else "NONE")

        inputs = self._prepare_inputs(input_text)
        device = self.device
        inputs = {k: v.to(device) for k,v in inputs.items()}

        # Robust forced_bos_token_id resolution using normalize_user_lang and various heuristics
        try:
            norm_tgt = normalize_user_lang(tgt_lang) if tgt_lang else tgt_lang
        except Exception:
            norm_tgt = tgt_lang

        set_forced = False
        if self.lang_code_map:
            # Determine desired short marker: prefer 'eng' for english, 'zho' for chinese, else take the first segment
            desired_short = None
            if isinstance(norm_tgt, str):
                # norm_tgt examples: 'eng_Latn', 'zho_Hans'
                desired_short = norm_tgt.split("_")[0].lower()
            if not desired_short and isinstance(tgt_lang, str):
                desired_short = tgt_lang.lower().split("_")[0]

            # Map common user short codes to tokenizer prefixes
            if desired_short in {"en", "eng"}:
                desired_short = "eng"
            if desired_short in {"zh", "cmn", "zho", "chinese"}:
                # use 'zho' prefix for Chinese canonical mapping
                desired_short = "zho"

            # Try exact keys and reasonable variants first
            try_keys = []
            if isinstance(norm_tgt, str):
                try_keys += [norm_tgt, norm_tgt.lower(), norm_tgt.replace('_','-'), norm_tgt.replace('-','_')]
                try_keys += [norm_tgt.split("_")[0] if "_" in norm_tgt else None]
            try_keys.append(tgt_lang)
            if isinstance(tgt_lang, str):
                try_keys += [tgt_lang.lower(), tgt_lang.replace('-', '_'), tgt_lang.replace('_','-')]

            # Try exact matches in lang_code_map
            for key in [k for k in try_keys if k]:
                if key in self.lang_code_map:
                    try:
                        kwargs["forced_bos_token_id"] = int(self.lang_code_map[key])
                        set_forced = True
                        logger.debug("Set forced_bos_token_id using exact key=%s", key)
                        break
                    except Exception:
                        continue

            # If not found, try prefix match using desired_short
            if not set_forced and desired_short:
                for key in self.lang_code_map:
                    if not isinstance(key, str):
                        continue
                    lkey = key.lower()
                    if lkey.startswith(desired_short):
                        try:
                            kwargs["forced_bos_token_id"] = int(self.lang_code_map[key])
                            set_forced = True
                            logger.debug("Set forced_bos_token_id using prefix match key=%s for desired_short=%s", key, desired_short)
                            break
                        except Exception:
                            continue

            # If still not found, attempt contains match (less strict)
            if not set_forced and desired_short:
                for key in self.lang_code_map:
                    try:
                        if desired_short in str(key).lower():
                            kwargs["forced_bos_token_id"] = int(self.lang_code_map[key])
                            set_forced = True
                            logger.debug("Set forced_bos_token_id using contains match key=%s for desired_short=%s", key, desired_short)
                            break
                    except Exception:
                        continue

        if not set_forced:
            logger.debug("Could not set forced_bos_token_id for tgt_lang=%s (norm=%s). lang_code_map keys sample: %s",
                         tgt_lang, norm_tgt, list(self.lang_code_map.keys())[:20] if self.lang_code_map else "NONE")

        # Run generate
        with torch.no_grad():
            gen_out = self.model.generate(**inputs, **kwargs)

        out = {}
        try:
            seqs = getattr(gen_out, "sequences", None)
            if seqs is not None:
                seq = seqs[0].cpu().numpy().tolist()
                out["sequences_ids"] = seq
                out["translated_text"] = self.tokenizer.decode(seq, skip_special_tokens=True)
            else:
                if isinstance(gen_out, torch.Tensor):
                    seq = gen_out[0].cpu().numpy().tolist()
                    out["sequences_ids"] = seq
                    out["translated_text"] = self.tokenizer.decode(seq, skip_special_tokens=True)
                else:
                    out["translated_text"] = ""
                    out["sequences_ids"] = []
        except Exception as e:
            logger.debug("Decoding sequences error: %s", e)
            out["translated_text"] = ""
            out["sequences_ids"] = []

        out["scores"] = getattr(gen_out, "scores", None)
        out["cross_attentions"] = getattr(gen_out, "cross_attentions", None)
        out["model_id"] = self.model_name
        out["inputs_ids"] = inputs.get("input_ids").cpu().numpy().tolist()
        out["input_len"] = int(inputs.get("input_ids").shape[1])
        return out

    def translate_async(self, *args, **kwargs):
        return self.executor.submit(self.translate_sync, *args, **kwargs)

# -------------------------
# AlignmentExtractor (robust)
# -------------------------
class AlignmentExtractor:
    def __init__(self, attn_threshold:float=0.18):
        self.attn_threshold = float(attn_threshold)

    def _to_numpy(self, x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        try:
            return np.array(x, dtype=float)
        except Exception:
            return None

    def compute_per_step_attentions(self, cross_attentions, input_len:int):
        """
        Convert many possible HF cross_attentions shapes into a list of 1D arrays (one per decode step).
        Each array length == input_len (src tokens).
        """
        if cross_attentions is None:
            return None
        try:
            # Convert to numpy if possible
            arr = self._to_numpy(cross_attentions)
            if arr is not None:
                # Cases by ndim
                if arr.ndim == 4:
                    # (layers, heads, tgt_len, src_len)
                    L, H, T, S = arr.shape
                    per_step = []
                    for t in range(T):
                        # sum over layers & heads for step t
                        step = arr[:, :, t, :].sum(axis=(0,1))
                        if step.sum() > 0: step = step / step.sum()
                        per_step.append(step)
                    return per_step
                elif arr.ndim == 3:
                    # (layers, tgt_len, src_len) or (tgt_len, src_len, ???) unknown
                    # assume (layers, tgt_len, src_len)
                    L, T, S = arr.shape
                    per_step = []
                    for t in range(T):
                        step = arr[:, t, :].sum(axis=0)
                        if step.sum() > 0: step = step / step.sum()
                        per_step.append(step)
                    return per_step
                elif arr.ndim == 2:
                    # (tgt_len, src_len)
                    per_step = []
                    for t in range(arr.shape[0]):
                        step = arr[t]
                        if step.sum() > 0: step = step / step.sum()
                        per_step.append(step)
                    return per_step
                elif arr.ndim == 1:
                    # single vector over src_len
                    vec = arr
                    if vec.size != input_len:
                        # pad or trim
                        if vec.size < input_len:
                            pad = np.zeros((input_len - vec.size,))
                            vec = np.concatenate([vec, pad])
                        else:
                            vec = vec[:input_len]
                    if vec.sum() > 0: vec = vec / vec.sum()
                    return [vec]
                else:
                    # unexpected shape: try reducing last dim
                    sdim = arr.shape[-1]
                    if sdim == input_len:
                        if arr.ndim >= 2:
                            possible_steps = arr.shape[-2]
                            per_step = []
                            for t in range(possible_steps):
                                step = arr[..., t, :].sum(axis=tuple(range(arr.ndim-2)))
                                if step.sum() > 0: step = step / step.sum()
                                per_step.append(step)
                            return per_step
                    return None

            if isinstance(cross_attentions, (list, tuple)):
                first = cross_attentions[0]
                try:
                    la = [self._to_numpy(x) for x in cross_attentions]
                    if la and la[0] is not None and la[0].ndim == 3:
                        H, T, S = la[0].shape
                        per_step = []
                        for t in range(T):
                            step = np.stack([l[:, t, :].sum(axis=0) for l in la], axis=0).sum(axis=0)
                            if step.sum() > 0: step = step / step.sum()
                            per_step.append(step)
                        return per_step
                except Exception:
                    pass

                try:
                    if isinstance(first, (list, tuple)):
                        per_step = []
                        for step_elem in cross_attentions:
                            layer_arrays = [self._to_numpy(l) for l in step_elem]
                            layer_sums = []
                            for la in layer_arrays:
                                if la is None:
                                    continue
                                la = np.asarray(la)
                                if la.ndim == 3:
                                    layer_sums.append(la.sum(axis=0)[-1])
                                elif la.ndim == 2:
                                    layer_sums.append(la.sum(axis=0))
                                elif la.ndim == 1:
                                    layer_sums.append(la)
                            if layer_sums:
                                step = np.mean(np.stack(layer_sums, axis=0), axis=0)
                            else:
                                step = np.zeros((input_len,))
                            if step.size != input_len:
                                if step.size < input_len:
                                    step = np.concatenate([step, np.zeros((input_len - step.size,))])
                                else:
                                    step = step[:input_len]
                            if step.sum() > 0: step = step / step.sum()
                            per_step.append(step)
                        return per_step
                except Exception:
                    pass

            return None
        except Exception as e:
            logger.exception("Alignment compute error: %s", e)
            return None

    def extract_alignments(self, cross_attentions, input_len:int, tokenizer, src_tokens: List[Token], sequences_ids:List[int]):
        per_step_attn = self.compute_per_step_attentions(cross_attentions, input_len)
        if per_step_attn is None:
            return [], []
        attns = []
        for a in per_step_attn:
            arr = np.asarray(a) if a is not None else np.zeros((input_len,))
            if arr.ndim == 0:
                arr = np.asarray([float(arr)])
            if arr.size < input_len:
                pad = np.zeros((input_len - arr.size,))
                arr = np.concatenate([arr, pad])
            elif arr.size > input_len:
                arr = arr[:input_len]
            s = float(arr.sum()) if float(arr.sum()) != 0 else 1.0
            arr = arr / s
            attns.append(arr)
        target_token_aligned_src_idxs = []
        for att in attns:
            aligned = [int(i) for i,x in enumerate(att.tolist()) if isinstance(x, (int,float)) and x >= self.attn_threshold]
            if not aligned:
                aligned = [int(np.argmax(att))] if att.size>0 else []
            target_token_aligned_src_idxs.append(aligned)
        source_to_target = [[] for _ in range(len(src_tokens))]
        for tgt_idx, aligned_idxs in enumerate(target_token_aligned_src_idxs):
            for sidx in aligned_idxs:
                if 0 <= sidx < len(src_tokens):
                    source_to_target[sidx].append(tgt_idx)
        return source_to_target, target_token_aligned_src_idxs

# -------------------------
# Confidence calculator
# -------------------------
class ConfidenceCalculator:
    def __init__(self, alpha:float=0.7, temperature:Optional[float]=None):
        self.alpha = alpha
        self.temperature = temperature

    def mt_probs_from_scores(self, scores):
        mt_probs = []
        if scores is None:
            return mt_probs
        for s in scores:
            try:
                if isinstance(s, torch.Tensor):
                    logits = s.cpu().numpy()
                else:
                    logits = np.array(s)
                if logits.ndim == 1:
                    if self.temperature and self.temperature > 0:
                        logits = logits / float(self.temperature)
                    probs = softmax_numpy(logits)
                    mt_probs.append(float(np.max(probs)))
                else:
                    logits = logits.flatten()
                    if self.temperature and self.temperature > 0:
                        logits = logits / float(self.temperature)
                    probs = softmax_numpy(logits)
                    mt_probs.append(float(np.max(probs)))
            except Exception:
                try:
                    import torch as _torch
                    probs = _torch.softmax(s, dim=-1)
                    mt_probs.append(float(_torch.max(probs).item()))
                except Exception:
                    mt_probs.append(0.0)
        return mt_probs

    def fuse(self, mt_probs:List[float], target_to_src_align:List[List[int]], src_tokens:List[Token], alpha:Optional[float]=None):
        alpha = self.alpha if alpha is None else alpha
        token_confs = []
        for i, mt_p in enumerate(mt_probs):
            aligned_src = target_to_src_align[i] if i < len(target_to_src_align) else []
            asr_values = []
            for s in aligned_src:
                if 0 <= s < len(src_tokens):
                    v = src_tokens[s].asr_conf
                    if v is not None:
                        asr_values.append(v)
            if asr_values:
                asr_mean = float(sum(asr_values)/len(asr_values))
            else:
                all_vals = [t.asr_conf for t in src_tokens if t.asr_conf is not None]
                asr_mean = float(sum(all_vals)/len(all_vals)) if all_vals else 0.5
            fused = alpha * float(mt_p) + (1.0 - alpha) * asr_mean
            token_confs.append(clamp(fused))
        chunk_conf = float(sum(token_confs)/len(token_confs)) if token_confs else 0.0
        return token_confs, clamp(chunk_conf)

# -------------------------
# Post-normalizer & TTS builder
# -------------------------
PUNCT_PAUSE_MS = {
    ".": (350, 600),
    ",": (150, 300),
    ";": (250, 400),
    "—": (250, 400),
}

def choose_pause_ms(punct: str):
    rng = PUNCT_PAUSE_MS.get(punct, (150,300))
    return int((rng[0] + rng[1]) / 2)

class PostNormalizer:
    def normalize_for_tts(self, translated_text:str, target_words:List[str]):
        tts_text = translated_text.strip()
        pause_hints = []
        for i,w in enumerate(target_words):
            if re.search(r'[.!?]$', w):
                pause_hints.append({"token_index": i, "pause_ms": choose_pause_ms(".")})
            elif w.endswith(","):
                pause_hints.append({"token_index": i, "pause_ms": choose_pause_ms(",")})
        ssml_parts = []
        for i,w in enumerate(target_words):
            clean_w = re.sub(r"[,\.!?;:]+$", "", w)
            ssml_parts.append(clean_w)
            ph = next((p for p in pause_hints if p["token_index"]==i), None)
            if ph:
                ssml_parts.append(f"<break time=\"{ph['pause_ms']}ms\"/>")
        ssml = "<speak>" + " ".join(ssml_parts) + "</speak>"
        return tts_text, ssml, pause_hints

    def build_tts_chunk(self, session_id, chunk_id, translated_text, tts_text, ssml,
                        source_words, target_words, mapping, token_confidences, chunk_conf,
                        model_id, cache_hit, prosody, metadata, target_word_timestamps=None,
                        replace_chunk_id=None, partial=False):
        return TTSChunk(
            session_id=session_id,
            chunk_id=chunk_id,
            translated_text=translated_text,
            tts_text=tts_text,
            ssml=ssml,
            source_words=source_words,
            target_words=target_words,
            word_mapping=mapping,
            target_word_timestamps=target_word_timestamps,
            pause_hints=metadata.get("pause_hints", []),
            token_confidences=token_confidences,
            confidence=chunk_conf,
            model_id=model_id,
            cache_hit=cache_hit,
            prosody=prosody,
            metadata=metadata,
            replace_chunk_id=replace_chunk_id,
            partial=partial
        )

# -------------------------
# Orchestrator
# -------------------------
class MTForTTS:
    def __init__(self,
                 mt_model_preference:List[str]=None,
                 device:Optional[str]=None,
                 cache_maxsize:int=4096,
                 context_chunks:int=2,
                 overlap_ms:int=250,
                 pause_threshold_ms:int=600,
                 attn_threshold:float=0.18,
                 alpha_conf:float=0.7,
                 num_workers:int=2):
        self.preproc = Preprocessor()
        self.chunk_mgr = ChunkManager(pause_threshold_ms=pause_threshold_ms, overlap_ms=overlap_ms)
        self.punct_tru = PunctuatorTruecaserNormalizer()
        self.context_mgr = ContextManager(context_chunks=context_chunks, cache_maxsize=cache_maxsize)
        self.mt = MTService(model_name_preference=mt_model_preference or ["facebook/nllb-200-1.3B", "facebook/nllb-200-distilled-600M"], device=device, max_workers=max(1,num_workers))
        self.aligner = AlignmentExtractor(attn_threshold=attn_threshold)
        self.confcalc = ConfidenceCalculator(alpha=alpha_conf)
        self.postnorm = PostNormalizer()
        self.idiom = IdiomHandler()
        self.executor = ThreadPoolExecutor(max_workers=max(1,num_workers))
        self.min_chars_for_mt = 1

    def process_chunk(self, chunk_msg:ChunkMessage, partial_mode:bool=False, replace_chunk_id:Optional[int]=None):
        t0 = now_s()

        # Normalize language codes right away to canonical NLLB-like codes
        try:
            chunk_msg.src_lang = normalize_user_lang(chunk_msg.src_lang)
        except Exception:
            pass
        try:
            chunk_msg.tgt_lang = normalize_user_lang(chunk_msg.tgt_lang)
        except Exception:
            pass

        # expand idioms to improve MT (pre-MT) -- pass both src and tgt language codes
        chunk_msg.raw_text = self.idiom.expand(chunk_msg.raw_text, chunk_msg.src_lang, chunk_msg.tgt_lang)

        pp_tokens, pp_text = self.preproc.preprocess_chunk(chunk_msg)
        if not pp_tokens or len(pp_text.strip()) < self.min_chars_for_mt:
            logger.debug("Chunk too short or empty after preprocessing.")
            return None

        # if source is Chinese, and tokens are whitespace-joined, fallback to char tokens for better mapping
        if chunk_msg.src_lang.startswith("zho") or chunk_msg.src_lang.startswith("zh"):
            # if incoming tokens appear as whole sentence, rebuild char tokens
            if len(pp_tokens) == 1 and len(pp_tokens[0].text) > 4:
                chars = []
                base_start = chunk_msg.chunk_start_ms
                total_chars = len(pp_tokens[0].text)
                per = max(20, int((chunk_msg.chunk_end_ms - chunk_msg.chunk_start_ms) / max(1,total_chars)))
                for i,ch in enumerate(list(pp_tokens[0].text)):
                    chars.append(Token(text=ch, start_ms=base_start + i*per, end_ms=base_start + (i+1)*per, asr_conf=pp_tokens[0].asr_conf, orig_index=i))
                pp_tokens = chars
                pp_text = "".join([c.text for c in pp_tokens])

        chunk_msg.tokens = pp_tokens
        chunk_msg.raw_text = pp_text
        chunk_msg = self.chunk_mgr.attach_overlap(chunk_msg)

        punct_text, mapping_punct_to_orig = self.punct_tru.punctuate_truecase_normalize(chunk_msg, pp_tokens, chunk_msg.is_final)
        context_text = self.context_mgr.get_context()
        cache_key = self.context_mgr.make_cache_key(context_text, punct_text, chunk_msg.src_lang, chunk_msg.tgt_lang)

        # log cache key for debugging (helps find stale Hungarian hits)
        logger.debug("process_chunk cache_key=%s", cache_key)

        cached = self.context_mgr.cache_get(cache_key)
        if cached:
            cached_copy = copy.deepcopy(cached)
            cached_copy.cache_hit = True
            cached_copy.metadata["timing_ms"] = int((now_s() - t0) * 1000)
            if chunk_msg.is_final:
                self.chunk_mgr.finalize_and_record(chunk_msg)
                self.context_mgr.push_finalized_chunk(chunk_msg)
            # write outputs
            write_outputs_to_disk(cached_copy)
            return cached_copy

        gen_kwargs = {}
        if partial_mode:
            gen_kwargs["num_beams"] = 1
        gen_kwargs["max_length"] = max(64, min(512, len(pp_text.split()) * 3 + 10))

        mt_future = self.mt.translate_async(punct_text, chunk_msg.src_lang, chunk_msg.tgt_lang, context=context_text, gen_kwargs=gen_kwargs)
        mt_result = mt_future.result()
        translated_text = mt_result.get("translated_text", "")
        model_id = mt_result.get("model_id", self.mt.model_name)

        source_to_target, target_to_src = self.aligner.extract_alignments(mt_result.get("cross_attentions"), mt_result.get("input_len",0), self.mt.tokenizer, pp_tokens, mt_result.get("sequences_ids", []))
        mt_probs = self.confcalc.mt_probs_from_scores(mt_result.get("scores"))

        # target words: for Chinese prefer not to split on whitespace (the decoded text may be no spaces)
        if chunk_msg.tgt_lang.startswith("zho") or chunk_msg.tgt_lang.startswith("zh"):
            # simple heuristic: if tokenizer already separates by spaces, keep; else split into characters
            if " " in translated_text.strip():
                target_words = translated_text.split()
            else:
                target_words = list(translated_text.strip())
        else:
            target_words = translated_text.split()

        # align mt_probs length to target_words
        if len(mt_probs) < len(target_words):
            pad_value = mt_probs[-1] if mt_probs else 0.5
            mt_probs = mt_probs + [pad_value] * (len(target_words) - len(mt_probs))
        if len(mt_probs) > len(target_words):
            mt_probs = mt_probs[:len(target_words)]

        token_confidences, chunk_conf = self.confcalc.fuse(mt_probs, target_to_src, pp_tokens)

        # derive timestamps for target words by mapping aligned src tokens
        tgt_to_src_map = {}
        for sidx, tgt_list in enumerate(source_to_target):
            for t in tgt_list:
                tgt_to_src_map.setdefault(t, []).append(sidx)
        target_word_timestamps = []
        for t_idx in range(len(target_words)):
            s_idxs = tgt_to_src_map.get(t_idx, [])
            if s_idxs:
                starts = [pp_tokens[s].start_ms for s in s_idxs if pp_tokens[s].start_ms is not None]
                ends = [pp_tokens[s].end_ms for s in s_idxs if pp_tokens[s].end_ms is not None]
                if starts and ends:
                    target_word_timestamps.append((int(min(starts)), int(max(ends))))
                else:
                    target_word_timestamps.append((chunk_msg.chunk_start_ms, chunk_msg.chunk_end_ms))
            else:
                target_word_timestamps.append((chunk_msg.chunk_start_ms, chunk_msg.chunk_end_ms))

        tts_text, ssml, pause_hints = self.postnorm.normalize_for_tts(translated_text, target_words)
        metadata = {"model_id": model_id, "gen_time_s": now_s() - t0, "pause_hints": pause_hints}

        source_words = [p.text for p in pp_tokens]
        tts_chunk = self.postnorm.build_tts_chunk(
            session_id=chunk_msg.session_id,
            chunk_id=chunk_msg.chunk_id,
            translated_text=translated_text,
            tts_text=tts_text,
            ssml=ssml,
            source_words=source_words,
            target_words=target_words,
            mapping=source_to_target,
            token_confidences=token_confidences,
            chunk_conf=chunk_conf,
            model_id=model_id,
            cache_hit=False,
            prosody={},
            metadata=metadata,
            target_word_timestamps=target_word_timestamps,
            replace_chunk_id=replace_chunk_id,
            partial=partial_mode
        )

        # cache result
        self.context_mgr.cache_set(cache_key, tts_chunk)

        if chunk_msg.is_final:
            self.chunk_mgr.finalize_and_record(chunk_msg)
            self.context_mgr.push_finalized_chunk(chunk_msg)

        tts_chunk.metadata["timing_ms"] = int((now_s() - t0) * 1000)

        # write outputs for TTS ingestion
        write_outputs_to_disk(tts_chunk)

        return tts_chunk

# Helpers: language mapping (improved & robust)
LANG_MAP = {
    "en": "eng_Latn",
    "english": "eng_Latn",
    "eng_latn": "eng_Latn",
    # Corrected Chinese mappings to zho_Hans/zho_Hant
    "zh": "zho_Hans",
    "chinese": "zho_Hans",
    "zho": "zho_Hans",
    "zho_hans": "zho_Hans",
    "zh_hans": "zho_Hans",
    "zho_hant": "zho_Hant",
    "zh_hant": "zho_Hant",
    "zh-cn": "zho_Hans",
    "zh-sg": "zho_Hans",
    "zh-tw": "zho_Hant",
    "zh-hk": "zho_Hant",
    "cmn": "zho_Hans",
    "cmn_hans": "zho_Hans",
    "cmn_hant": "zho_Hant",
}

def normalize_user_lang(s: str) -> str:
    """
    Normalize a user-provided language string to a canonical NLLB-like code.
    - Accepts short names ("en", "zh"), BCP-47-like ("en-US", "zh-CN"), and some spellings ("english","chinese").
    - Returns canonical codes used in this pipeline (e.g. "eng_Latn", "zho_Hans") when recognizable,
      otherwise returns the normalized token (lowercased, separators to underscore) so the caller can decide.
    """
    if not s:
        return ""
    s_raw = s.strip()
    s_norm = s_raw.lower().replace(" ", "").replace("-", "_")
    # direct map
    if s_norm in LANG_MAP:
        return LANG_MAP[s_norm]
    # prefix matching (e.g. "en_us" -> "eng_Latn")
    if s_norm.startswith("en"):
        return "eng_Latn"
    # Chinese variants: prefer Hant for traditional indicators else default to Hans
    if s_norm.startswith("zh") or s_norm.startswith("zho") or s_norm.startswith("cmn"):
        # explicit Hant indicators
        if "hant" in s_norm or "tw" in s_norm or "hk" in s_norm:
            return "zho_Hant"
        # default to simplified (Hans)
        return "zho_Hans"
    # fallback: return the normalized form (caller can treat as unknown)
    # try to map common full nllb code if user provided it already (some users might type "eng_Latn")
    s_norm_underscore = s_norm.replace("-", "_")
    # e.g., user passed "eng_latn" or "zho_hans"
    if s_norm_underscore in LANG_MAP:
        return LANG_MAP[s_norm_underscore]
    return s_norm_underscore


# -------------------------
# Interactive demo
# -------------------------
def pretty_print_ttschunk(t: TTSChunk):
    d = {
        "session_id": t.session_id,
        "chunk_id": t.chunk_id,
        "translated_text": t.translated_text,
        "tts_text": t.tts_text,
        "ssml": t.ssml,
        "source_words": t.source_words,
        "target_words": t.target_words,
        "word_mapping": t.word_mapping,
        "target_word_timestamps": t.target_word_timestamps,
        "pause_hints": t.metadata.get("pause_hints", []),
        "token_confidences": t.token_confidences,
        "confidence": t.confidence,
        "model_id": t.model_id,
        "cache_hit": t.cache_hit,
        "timing_ms": t.metadata.get("timing_ms")
    }
    print(json.dumps(d, indent=2, ensure_ascii=False))

def interactive_cli():
    print("MT → TTS Interactive Demo (EN <-> ZH)\n")
    print("You will be asked to enter source language and target language.")
    print("Supported: 'en'/'english' and 'zh'/'chinese'. Type 'quit' at any prompt to exit.\n")

    device_pref = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    try:
        pipeline = MTForTTS(mt_model_preference=["facebook/nllb-200-1.3B", "facebook/nllb-200-distilled-600M"], device=device_pref, num_workers=1)
    except Exception as e:
        logger.exception("Failed to initialize pipeline: %s", e)
        print("Pipeline init failed. See log. Exiting.")
        return

    session_id = "demo_session"
    chunk_id = 1
    while True:
        src_lang_input = input("Enter source language (en / zh) or 'quit': ").strip()
        if not src_lang_input:
            continue
        if src_lang_input.lower() in {"quit","exit"}:
            print("Goodbye.")
            break
        tgt_lang_input = input("Enter target language (en / zh) or 'quit': ").strip()
        if not tgt_lang_input:
            continue
        if tgt_lang_input.lower() in {"quit","exit"}:
            print("Goodbye.")
            break

        src_code = normalize_user_lang(src_lang_input)
        tgt_code = normalize_user_lang(tgt_lang_input)
        print(f"Selected: {src_code} -> {tgt_code}")

        user_input = input(f"Enter text in {src_lang_input}: ").strip()
        if not user_input:
            print("Empty input, skipping.")
            continue
        if user_input.lower() in {"quit","exit"}:
            print("Goodbye.")
            break

        # construct tokens with timestamps (rough)
        if src_code.startswith("zho") or src_code.startswith("zh"):
            # char-level tokens for Chinese input to create mapping/timestamps
            tokens = []
            per = 50
            for i,ch in enumerate(list(user_input)):
                if ch.isspace():
                    continue
                tokens.append(Token(text=ch, start_ms=i*per, end_ms=(i+1)*per, asr_conf=0.95, orig_index=i))
            raw_text = "".join([t.text for t in tokens])
            chunk_end = len(tokens)*50 if tokens else 0
        else:
            words = user_input.split()
            tokens = [Token(text=w, start_ms=i*50, end_ms=(i+1)*50, asr_conf=0.95, orig_index=i) for i,w in enumerate(words)]
            raw_text = " ".join([t.text for t in tokens])
            chunk_end = len(tokens)*50

        chunk = ChunkMessage(
            session_id=session_id,
            chunk_id=chunk_id,
            is_final=True,
            src_lang=src_code,
            tgt_lang=tgt_code,
            chunk_start_ms=0,
            chunk_end_ms=chunk_end,
            tokens=tokens,
            raw_text=raw_text
        )

        try:
            out = pipeline.process_chunk(copy.deepcopy(chunk))
            if out:
                print("\n--- MT & TTS-ready output (saved to mt_outputs/) ---")
                pretty_print_ttschunk(out)
            else:
                print("No output (filtered / too short).")
        except Exception as e:
            logger.exception("Translation error: %s", e)
            print("Error during translation. See logs.")

        chunk_id += 1

if __name__ == "__main__":
    interactive_cli()
