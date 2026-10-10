"""Versioned skill library.

A skill is a short, reusable piece of procedural guidance ("when the question
asks for a ratio, CAST to REAL before dividing"). Every edit creates a new
immutable *version*; the library holds a pointer to the active version of
each skill. Version records keep their parent ids, so the lineage
(create -> update -> merge -> discard) can be replayed and diffed.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Skill:
    id: str                      # stable lineage id, e.g. "S007"
    name: str
    description: str             # when to use it (used for retrieval)
    content: str                 # the procedure itself (injected into the prompt)
    tags: list[str] = field(default_factory=list)
    version: int = 1
    parents: list[str] = field(default_factory=list)   # version keys "S003@v2"
    origin: str = "create"       # create | update | merge
    created_round: int = 0

    @property
    def key(self) -> str:
        return f"{self.id}@v{self.version}"

    def text(self) -> str:
        return f"[{self.name}]\nWhen to use: {self.description}\n{self.content}"

    def fingerprint(self) -> str:
        return hashlib.sha1(self.text().encode()).hexdigest()[:10]


class SkillLibrary:
    def __init__(self, skills: dict[str, Skill] | None = None, next_id: int = 1):
        self.skills: dict[str, Skill] = skills or {}
        self.next_id = next_id

    # ---- queries -------------------------------------------------------
    def active(self) -> list[Skill]:
        return list(self.skills.values())

    def get(self, sid: str) -> Skill | None:
        return self.skills.get(sid)

    def fingerprint(self) -> str:
        return hashlib.sha1("|".join(sorted(s.fingerprint() for s in self.active())).encode()).hexdigest()[:12]

    def __len__(self):
        return len(self.skills)

    def clone(self) -> "SkillLibrary":
        return SkillLibrary(copy.deepcopy(self.skills), self.next_id)

    # ---- edit operations (return the new Skill, or None for discard) ---
    def new_id(self) -> str:
        sid = f"S{self.next_id:03d}"
        self.next_id += 1
        return sid

    def create(self, name, description, content, tags=None, round_=0) -> Skill:
        s = Skill(self.new_id(), name, description, content, list(tags or []), created_round=round_)
        self.skills[s.id] = s
        return s

    def update(self, sid, round_=0, **changes) -> Skill:
        old = self.skills[sid]
        new = copy.deepcopy(old)
        for k, v in changes.items():
            if v:
                setattr(new, k, v)
        new.version = old.version + 1
        new.parents = [old.key]
        new.origin = "update"
        new.created_round = round_
        self.skills[sid] = new
        return new

    def merge(self, sids, name, description, content, tags=None, round_=0) -> Skill:
        parents = [self.skills[s].key for s in sids]
        for s in sids:
            del self.skills[s]
        s = Skill(self.new_id(), name, description, content, list(tags or []),
                  parents=parents, origin="merge", created_round=round_)
        self.skills[s.id] = s
        return s

    def discard(self, sid) -> None:
        del self.skills[sid]

    # ---- persistence ---------------------------------------------------
    def to_json(self) -> dict:
        return {"next_id": self.next_id, "skills": [asdict(s) for s in self.active()]}

    @classmethod
    def from_json(cls, d: dict) -> "SkillLibrary":
        skills = {x["id"]: Skill(**x) for x in d["skills"]}
        return cls(skills, d["next_id"])

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SkillLibrary":
        return cls.from_json(json.loads(Path(path).read_text()))


def text_diff(old: str, new: str) -> str:
    return "\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                         "before", "after", lineterm="", n=1))
