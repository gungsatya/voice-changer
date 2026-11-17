"""Parse emotion-tagged text such as

    [marah] Kenapa kamu terlambat lagi? [normal] Ya sudah, [ceria] ayo kita makan!

into segments [(emotion, text), ...]. A tag applies to everything after it until the next tag,
so a single word can be given its own emotion: "Tiang [ceria]seneng [normal]pisan."
Text before the first tag uses the default emotion.
"""

import re

TAG_RE = re.compile(r"\[\s*([A-Za-z][A-Za-z0-9_-]{0,31})\s*\]")


def parse(text: str, default_emotion: str = "normal") -> list[tuple[str, str]]:
    """Split tagged text into (emotion, text) segments; adjacent same-emotion segments are merged."""
    segments: list[tuple[str, str]] = []
    current = (default_emotion or "normal").strip().lower()
    pos = 0
    for m in TAG_RE.finditer(text or ""):
        _append(segments, current, text[pos:m.start()])
        current = m.group(1).lower()
        pos = m.end()
    _append(segments, current, (text or "")[pos:])
    return segments


def _append(segments, emotion, chunk):
    chunk = " ".join(chunk.split())
    if not chunk:
        return
    if segments and segments[-1][0] == emotion:
        segments[-1] = (emotion, f"{segments[-1][1]} {chunk}")
    else:
        segments.append((emotion, chunk))


def tag_each_word(text: str, default_emotion: str = "normal") -> str:
    """Rewrite text so every word carries an explicit tag, ready to be edited word by word.

    Words that already follow a tag keep that tag.
    """
    words = []
    for emotion, chunk in parse(text, default_emotion):
        words += [f"[{emotion}]{w}" for w in chunk.split()]
    return " ".join(words)


def unknown_emotions(segments, available) -> list[str]:
    available = set(available)
    return sorted({e for e, _ in segments if e not in available})
