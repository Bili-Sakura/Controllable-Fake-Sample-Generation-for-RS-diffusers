import json
import re
from pathlib import Path
from typing import Any, Dict


_COMMENT_RE = re.compile(r"//.*$")


def load_json_with_comments(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    cleaned = "\n".join(_COMMENT_RE.sub("", line) for line in text.splitlines())
    return json.loads(cleaned)
