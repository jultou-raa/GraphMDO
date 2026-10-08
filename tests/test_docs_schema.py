"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import json
from pathlib import Path

from mdo_framework.schema import StudySchema

SCHEMA_FILE = (
    Path(__file__).parent.parent / "docs" / "technical-reference" / "study-schema.json"
)
REGENERATE_COMMAND = (
    'uv run python -c "import json, pathlib; '
    "from mdo_framework.schema import StudySchema; "
    "pathlib.Path('docs/technical-reference/study-schema.json').write_text("
    "json.dumps(StudySchema.model_json_schema(), indent=2, sort_keys=True) + '\\n', "
    "encoding='utf-8', newline='\\n')\""
)


def render_json_schema() -> str:
    return json.dumps(StudySchema.model_json_schema(), indent=2, sort_keys=True) + "\n"


def test_published_json_schema_matches_the_models():
    published = SCHEMA_FILE.read_text(encoding="utf-8")

    assert published == render_json_schema(), (
        f"{SCHEMA_FILE.name} is out of date with mdo_framework.schema. "
        f"Regenerate it from the repository root with:\n{REGENERATE_COMMAND}"
    )
