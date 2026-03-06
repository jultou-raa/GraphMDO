from pathlib import Path
import re

HEADER = '''"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

'''

for path in list(Path("src").rglob("*.py")) + list(Path("tests").rglob("*.py")):
    text = path.read_text()
    if "Mozilla Public" not in text:
        path.write_text(HEADER + text)

print("Headers added")
