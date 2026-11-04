import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
notebook=json.loads((root/'Qwen3Merger.ipynb').read_text(encoding='utf-8'))
assert notebook['nbformat']==4
code_cells=[c for c in notebook['cells'] if c['cell_type']=='code']
for index,cell in enumerate(code_cells):compile(''.join(cell['source']),f'cell_{index}','exec')
tree=ast.parse(''.join(code_cells[0]['source']))
payload=None
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode':
        payload=base64.b64decode(ast.literal_eval(node.args[0]))
assert payload
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        assert archive.read(name)==(root/name).read_bytes(),f'Stale embedded source: {name}'
    assert 'src/trainer.py' in archive.namelist() and 'tests/test_reference_parity.py' in archive.namelist()
print(f'Validated {len(code_cells)} executable cells and {len(archive.namelist())} exact embedded files')
