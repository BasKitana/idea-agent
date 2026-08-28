from pathlib import Path

import config

# Zero-maintenance browsing surface using Obsidian's core Bases feature
# (no plugin dependency, unlike Dataview). Regenerated every run -- cheap,
# always current, and deliberately NOT a substitute for real links/MOCs:
# research is unambiguous that pre-built MOCs before real note volume is a
# top failure mode, so this is just a queryable table, no graph edges.

_VIEW_TEMPLATE = """  - type: table
    name: "{name}"
    filters:
      - 'file.inFolder("{folder}")'
    order:
      - file.name
      - tags
      - date
"""

_ALL_VIEW = """  - type: table
    name: "All"
    order:
      - file.name
      - type
      - tags
      - date
"""


_VIEW_NAMES = {"concept": "Concepts", "project": "Projects", "entity": "Entities", "log": "Logs"}


def write_index(vault_path: Path) -> Path:
    views = "".join(
        _VIEW_TEMPLATE.format(name=_VIEW_NAMES.get(note_type, note_type.capitalize()), folder=folder)
        for note_type, folder in config.FOLDER_BY_TYPE.items()
    )
    content = "views:\n" + views + _ALL_VIEW
    path = vault_path / "Index.base"
    path.write_text(content, encoding="utf-8")
    return path
