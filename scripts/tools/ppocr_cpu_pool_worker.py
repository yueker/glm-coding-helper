from __future__ import annotations

import json
import math
import os
import queue
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = os.environ.get("CNCAPTCHA_CPU_OCR_MODEL", "PP-OCRv6_tiny_rec")
ENGINE = os.environ.get("CNCAPTCHA_CPU_OCR_ENGINE", "paddle_dynamic")
WORKERS = int(os.environ.get("CNCAPTCHA_CPU_OCR_WORKERS", "3"))
CACHE_NAME = os.environ.get("CNCAPTCHA_CPU_PADDLEX_CACHE", ".paddlex_cache_cpu")
HYBRID_FAST_MODEL = os.environ.get("CNCAPTCHA_CPU_OCR_FAST_MODEL", "PP-OCRv5_mobile_rec")
HYBRID_FALLBACK_MODEL = os.environ.get("CNCAPTCHA_CPU_OCR_FALLBACK_MODEL", "PP-OCRv5_server_rec")
CONSTRAINED_DECODE = os.environ.get("CNCAPTCHA_CPU_OCR_CONSTRAINED", "1").lower() not in {
    "0",
    "false",
    "no",
}


def configure_env() -> None:
    paddle_home = ROOT / ".paddle_home"
    os.environ["HOME"] = str(paddle_home)
    os.environ["USERPROFILE"] = str(paddle_home)
    os.environ["PADDLE_HOME"] = str(paddle_home / ".cache" / "paddle")
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(ROOT / CACHE_NAME)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")


def first_cjk(text: str) -> str:
    return next((ch for ch in text if "\u4e00" <= ch <= "\u9fff"), "")


def predict_with_candidate_scores(recognizer, path: str, prompt: list[str]) -> dict:
    predictor = recognizer.paddlex_predictor
    raw_imgs = predictor.pre_tfs["Read"](imgs=[path])
    batch_imgs = predictor.pre_tfs["ReisizeNorm"](imgs=raw_imgs)
    x = predictor.pre_tfs["ToBatch"](imgs=batch_imgs)
    batch_preds = predictor.runner(x=x)
    probs = np.array(batch_preds[0] if isinstance(batch_preds, (list, tuple)) else batch_preds)
    texts, scores = predictor.post_op(batch_preds)

    candidate_scores: dict[str, float] = {}
    for char in prompt:
        idx = predictor.post_op.dict.get(char)
        if idx is None:
            candidate_scores[char] = 0.0
        else:
            candidate_scores[char] = float(probs[0, :, idx].max())

    best_char = max(candidate_scores, key=candidate_scores.get) if candidate_scores else first_cjk(str(texts[0]))
    return {
        "text": str(texts[0]),
        "char": best_char,
        "score": float(candidate_scores.get(best_char, scores[0] if scores else 0.0) or 0.0),
        "ocr_text": str(texts[0]),
        "ocr_score": float(scores[0] if scores else 0.0),
        "candidate_scores": candidate_scores,
    }


def recognizer_worker(req_q: Queue, resp_q: Queue, worker_id: int, model_name: str) -> None:
    configure_env()
    from paddleocr import TextRecognition

    recognizer = TextRecognition(model_name=model_name, device="cpu", engine=ENGINE)
    resp_q.put({"type": "ready", "worker": worker_id})
    while True:
        item = req_q.get()
        if item is None:
            break
        req_id, idx, path, prompt = item
        try:
            started = time.perf_counter()
            if CONSTRAINED_DECODE and prompt:
                row = predict_with_candidate_scores(recognizer, str(path), list(prompt))
            else:
                result = recognizer.predict(str(path))
                obj = result[0] if result else {}
                text = str(obj.get("rec_text", "")) if isinstance(obj, dict) else str(obj)
                score = float(obj.get("rec_score", 0.0) or 0.0) if isinstance(obj, dict) else 0.0
                row = {"text": text, "char": first_cjk(text), "score": score}
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            resp_q.put(
                {
                    "type": "result",
                    "req_id": req_id,
                    "idx": idx,
                    **row,
                    "elapsed_ms": elapsed_ms,
                    "worker": worker_id,
                }
            )
        except Exception as exc:
            resp_q.put({"type": "result", "req_id": req_id, "idx": idx, "error": str(exc), "worker": worker_id})
    recognizer.close()


def repair_to_prompt(pred_chars: list[str], prompt_chars: list[str]) -> tuple[list[str], bool]:
    repaired = list(pred_chars)
    in_prompt = [ch for ch in repaired if ch in prompt_chars]
    missing = [ch for ch in prompt_chars if ch not in in_prompt]
    bad_idxs = [idx for idx, ch in enumerate(repaired) if ch not in prompt_chars]
    if len(missing) == len(bad_idxs) == 1:
        repaired[bad_idxs[0]] = missing[0]
    return repaired, repaired != pred_chars


def can_map_prompt(rows: list[dict], prompt: list[str]) -> bool:
    if not prompt:
        return True
    chars = [str(row.get("char", "")) for row in rows]
    repaired, _ = repair_to_prompt(chars, prompt)
    used: set[int] = set()
    for prompt_ch in prompt:
        found = None
        for idx, box_ch in enumerate(repaired):
            if idx not in used and box_ch == prompt_ch:
                found = idx
                break
        if found is None:
            return False
        used.add(found)
    return True


def assign_prompt_globally(rows: list[dict], prompt: list[str]) -> list[dict]:
    if len(rows) != len(prompt):
        return rows

    best_perm: tuple[str, ...] | None = None
    best_score = -float("inf")

    def permutations(items: list[str]):
        if len(items) <= 1:
            yield tuple(items)
            return
        for idx, item in enumerate(items):
            rest = items[:idx] + items[idx + 1 :]
            for suffix in permutations(rest):
                yield (item,) + suffix

    for perm in permutations(list(prompt)):
        score = 0.0
        for row, char in zip(rows, perm):
            candidate_scores = row.get("candidate_scores") or {}
            prob = float(candidate_scores.get(char, 0.0) or 0.0)
            score += math.log(max(prob, 1e-12))
        if score > best_score:
            best_score = score
            best_perm = perm

    if best_perm is None:
        return rows

    assigned = []
    for row, char in zip(rows, best_perm):
        updated = dict(row)
        updated["raw_char"] = updated.get("char", "")
        updated["char"] = char
        updated["score"] = float((updated.get("candidate_scores") or {}).get(char, updated.get("score", 0.0)) or 0.0)
        assigned.append(updated)
    return assigned


class CpuOcrPool:
    def __init__(self, workers: int, model_name: str) -> None:
        self.model_name = model_name
        self.req_q: Queue = Queue()
        self.resp_q: Queue = Queue()
        self.procs = [
            Process(target=recognizer_worker, args=(self.req_q, self.resp_q, idx, model_name))
            for idx in range(workers)
        ]
        for proc in self.procs:
            proc.start()
        ready = 0
        while ready < workers:
            msg = self.resp_q.get(timeout=120)
            if msg.get("type") == "ready":
                ready += 1
        sys.stderr.write(f"[ppocr-cpu-pool] {model_name} {ENGINE} workers={workers} loaded\n")
        sys.stderr.flush()

    def predict(self, req_id: int, paths: list[str], prompt: list[str], timeout: float = 30.0) -> list[dict]:
        for idx, path in enumerate(paths):
            self.req_q.put((req_id, idx, path, prompt))

        deadline = time.perf_counter() + timeout
        results: list[dict] = []
        while len(results) < len(paths):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("PP-OCR CPU pool timeout")
            try:
                msg = self.resp_q.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("PP-OCR CPU pool timeout") from exc
            if msg.get("type") == "result" and msg.get("req_id") == req_id:
                results.append(msg)

        results.sort(key=lambda item: int(item["idx"]))
        for item in results:
            if item.get("error"):
                raise RuntimeError(str(item["error"]))
        return [
            {
                "text": str(item.get("text", "")),
                "char": str(item.get("char", "")),
                "score": float(item.get("score", 0.0) or 0.0),
                "ocr_text": str(item.get("ocr_text", item.get("text", ""))),
                "ocr_score": float(item.get("ocr_score", item.get("score", 0.0)) or 0.0),
                "candidate_scores": item.get("candidate_scores", {}),
                "elapsed_ms": round(float(item.get("elapsed_ms", 0.0) or 0.0), 1),
                "worker": int(item.get("worker", -1)),
            }
            for item in results
        ]

    def close(self) -> None:
        for _ in self.procs:
            self.req_q.put(None)
        for proc in self.procs:
            proc.join(timeout=5)


def main() -> int:
    configure_env()
    hybrid = MODEL_NAME.lower() in {"hybrid", "ppocrv5_hybrid", "mobile_server_hybrid"}
    if hybrid:
        fast_pool = CpuOcrPool(WORKERS, HYBRID_FAST_MODEL)
        fallback_pool = None
        pool = fast_pool
    else:
        fast_pool = fallback_pool = None
        pool = CpuOcrPool(WORKERS, MODEL_NAME)
    req_id = 0
    try:
        for raw_line in sys.stdin.buffer:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                req_id += 1
                paths = [str(path) for path in (req.get("paths") or [])]
                prompt = list(req.get("prompt") or req.get("chars") or [])
                rows = pool.predict(req_id, paths, prompt)
                if CONSTRAINED_DECODE and prompt and len(rows) == len(prompt):
                    rows = assign_prompt_globally(rows, prompt)
                used_model = HYBRID_FAST_MODEL if hybrid else MODEL_NAME
                fallback_used = False
                if hybrid and not can_map_prompt(rows, prompt):
                    if fallback_pool is None:
                        fallback_pool = CpuOcrPool(WORKERS, HYBRID_FALLBACK_MODEL)
                    req_id += 1
                    rows = fallback_pool.predict(req_id, paths, prompt)
                    if CONSTRAINED_DECODE and prompt and len(rows) == len(prompt):
                        rows = assign_prompt_globally(rows, prompt)
                    used_model = HYBRID_FALLBACK_MODEL
                    fallback_used = True
                sys.stdout.write(
                    json.dumps(
                        {
                            "success": True,
                            "results": rows,
                            "model": used_model,
                            "fallback_used": fallback_used,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                sys.stdout.flush()
            except Exception as exc:
                import traceback

                traceback.print_exc(file=sys.stderr)
                sys.stdout.write(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False) + "\n")
                sys.stdout.flush()
    finally:
        pool.close()
        if hybrid and fallback_pool is not None:
            fallback_pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
