from pathlib import Path

pyproject = Path("pyproject.toml")
text = pyproject.read_text()
text = text.replace('requires-python = ">=3.12"', 'license = {text = "MPL-2.0"}\nrequires-python = ">=3.12"')
pyproject.write_text(text)
print("done")
