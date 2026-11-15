import json
from pathlib import Path

cells = []
def cell(kind, text):
    c = dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True))
    if kind == 'code':
        c.update(outputs=[],execution_count=None)
    cells.append(c)

cell('markdown', '# Potsdam traditional baseline: RGB decision tree\nAttach this code folder and your existing final Potsdam dataset. CPU is sufficient. Training searches four candidates using validation only. Test evaluation is separate. No deep-model retraining is needed.')
cell('code', '''from pathlib import Path
import shutil, subprocess, sys, json
CODE_SOURCE = None  # Optional explicit folder containing this baseline run.py
DATA_ROOT = None    # Optional folder containing manifest.json (or manifest path)
if CODE_SOURCE is None:
    matches = [p.parent for p in Path('/kaggle/input').rglob('make_notebook.py')
               if (p.parent/'run.py').is_file() and 'CPU RGB decision tree' in (p.parent/'run.py').read_text()]
    if len(matches) != 1:
        raise RuntimeError(f'Set CODE_SOURCE explicitly. Matches: {matches}')
    CODE_SOURCE = matches[0]
if DATA_ROOT is None:
    matches = []
    for p in Path('/kaggle/input').rglob('manifest.json'):
        try:
            m = json.loads(p.read_text())
            if m.get('dataset') == 'Potsdam' and len(m.get('tiles', [])) == 38:
                matches.append(p.parent)
        except (ValueError, OSError):
            pass
    if len(matches) != 1:
        raise RuntimeError(f'Set DATA_ROOT explicitly. Matches: {matches}')
    DATA_ROOT = matches[0]
DATA_ROOT = Path(DATA_ROOT)
if DATA_ROOT.is_file():
    DATA_ROOT = DATA_ROOT.parent
assert (DATA_ROOT/'manifest.json').is_file(), DATA_ROOT
WORK = Path('/kaggle/working/traditional_baseline')
WORK.mkdir(exist_ok=True)
for name in ('run.py','boundary.py','requirements.txt'):
    shutil.copy2(Path(CODE_SOURCE)/name, WORK/name)
OUTPUT = Path('/kaggle/working/traditional_baseline_results')
print('Code:', CODE_SOURCE, '\\nData:', DATA_ROOT, '\\nWork:', WORK)
''')
cell('code', '''# Kaggle usually includes these packages. Install only if imports fail.
probe = subprocess.run([sys.executable, '-c', 'import numpy, sklearn, rasterio, joblib'])
if probe.returncode:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(WORK/'requirements.txt')], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(WORK/'environment.txt').open('w'), check=True)
def execute(*arguments):
    subprocess.run([sys.executable, '-u', str(WORK/'run.py'), *map(str,arguments)], cwd=WORK, check=True)
''')
cell('code', '''# 16 training tiles, 30,000 uniformly sampled pixels each; four validation trials.
# The loader automatically resolves nested TIFF folders and excludes 4_12 and 6_7.
execute('train', '--data', DATA_ROOT, '--output', OUTPUT/'training')
CHECKPOINT = OUTPUT/'training'/'best.joblib'
''')
cell('code', '''# Full validation metrics and prediction TIFFs, including exact boundary F1.
CHECKPOINT = OUTPUT/'training'/'best.joblib'
assert CHECKPOINT.is_file(), 'Set CHECKPOINT to your saved trusted best.joblib if using a new session.'
execute('evaluate', '--data', DATA_ROOT, '--checkpoint', CHECKPOINT,
        '--split', 'validation', '--output', OUTPUT/'validation')
''')
cell('code', '''# Run only after selection is complete. Test scores must not guide further tuning.
RUN_TEST = False  # Set True for the final held-out evaluation.
if RUN_TEST:
    execute('evaluate', '--data', DATA_ROOT, '--checkpoint', CHECKPOINT,
            '--split', 'test', '--output', OUTPUT/'test')
''')
cell('code', '''# Download this archive or save a notebook version with its outputs.
archive = shutil.make_archive('/kaggle/working/traditional_baseline_results', 'zip', OUTPUT)
print(archive)
''')
Path(__file__).with_name('kaggle_demo.ipynb').write_text(json.dumps(dict(cells=cells,
 metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
 nbformat=4,nbformat_minor=5),indent=2))
