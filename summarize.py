import re
from collections import Counter


def summarize_text(
    text: str,
    *,
    min_sentences: int = 3,
    max_sentences: int = 10,
    ratio: float = 0.2,
) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    sentences = _split_sentences(cleaned)
    if len(sentences) <= min_sentences:
        return cleaned

    tokens = _tokenize(cleaned)
    if not tokens:
        return cleaned

    freq = Counter(tokens)
    scores = []
    for idx, sent in enumerate(sentences):
        sent_tokens = _tokenize(sent)
        if not sent_tokens:
            continue
        score = sum(freq.get(t, 0) for t in sent_tokens) / len(sent_tokens)
        scores.append((score, idx, sent))

    if not scores:
        return cleaned

    target = int(max(min_sentences, min(max_sentences, round(len(sentences) * ratio))))
    top = sorted(scores, reverse=True)[:target]
    ordered = [item[2] for item in sorted(top, key=lambda x: x[1])]
    return "\n".join(ordered)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.!\?。！？])\s+", text)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences if sentences else [text.strip()]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|[\u3040-\u30ff\u3400-\u9fff]+", text)
