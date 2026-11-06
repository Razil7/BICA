"""Выполняет ноутбук. usage: run_nb.py in.ipynb out.ipynb [KEY=VALUE ...] — временная подмена параметров настройки.
Исходный код ячеек в сохранённом файле остаётся прежним; предупреждения из stderr убираются."""
import re, sys
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

src, dst, overrides = sys.argv[1], sys.argv[2], dict(a.split("=", 1) for a in sys.argv[3:])
nb = nbformat.read(src, as_version=4)
original = [c.source for c in nb.cells]
for c in nb.cells:
    if c.cell_type == "code":
        for k, v in overrides.items():
            c.source = re.sub(rf"(?m)^{k} = [^#\n]*", f"{k} = {v}  ", c.source)
try:
    NotebookClient(nb, timeout=7200, kernel_name="python3").execute()
except CellExecutionError as e:
    print("ОШИБКА ВЫПОЛНЕНИЯ:", str(e)[-1500:])
for c, s in zip(nb.cells, original):
    c.source = s
    if c.cell_type == "code":
        c.outputs = [o for o in c.outputs
                     if not (o.output_type == "stream" and o.name == "stderr" and "Warning" in o.text)]
nbformat.write(nb, dst)
for c in nb.cells:
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            print(re.sub(r"[^\n\r]*it/s\]|[^\n\r]*s/it\]|[^\n\r]*\?it/s\]", "", o.text), end="")
        elif o.output_type in ("execute_result", "display_data") and "text/plain" in o.data:
            print(o.data["text/plain"])
        elif o.output_type == "error":
            print("ERROR:", o.ename, o.evalue)
