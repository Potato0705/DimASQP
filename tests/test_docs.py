"""Keep the README snippet and the example script runnable."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_python_snippets_run():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", readme, re.S)
    assert len(blocks) == 2
    namespace = {}
    for block in blocks:  # the second snippet continues the first
        exec(block, namespace)
    assert namespace["out"]["va"].shape == (2, 2, 2)
    assert namespace["loss_va"].item() > 0
    assert namespace["token_va"].shape == (2, 16, 2)
    assert namespace["loss_pos"].item() > 0
    assert namespace["quad_va"].shape == (2, 2, 2)


def test_example_script_runs(capsys):
    spec = importlib.util.spec_from_file_location("attach_to_encoder", ROOT / "examples" / "attach_to_encoder.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    batch = example.synthetic_batch()
    for name in ("Position", "Span-Pair", "SP+Prior", "Opinion-Guided"):
        example.train(name, batch, steps=2)
    assert "predicted VA" in capsys.readouterr().out
