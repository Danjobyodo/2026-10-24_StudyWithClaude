import re
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

import os
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repository root

SETUP = r'''# ==== 環境設定（Google Colab / ローカル共通）====
# このノートブックは単独で実行できる（共通コードは下に埋め込まれており、他のファイルを読み込まない）。
# 出力（results/, figures/）はこのランタイムの作業ディレクトリに保存される。Colab ではランタイム終了で
# 消えるので、残したい場合は Google Drive を使う：
#   from google.colab import drive; drive.mount('/content/drive')
#   os.makedirs('/content/drive/MyDrive/D4_vs_MSD', exist_ok=True); os.chdir('/content/drive/MyDrive/D4_vs_MSD')
import os, sys, subprocess, json
IN_COLAB = 'google.colab' in sys.modules
RESULTS, FIGURES = 'results', 'figures'
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
SEED = 20261024
print('cwd =', os.getcwd(), '| Colab =', IN_COLAB)'''

INSTALL = r'''# ==== 外部ライブラリ（Colab に無いものだけをインストール）====
import importlib.util
for _mod, _pkg in [('stim', 'stim'), ('scipy', 'scipy'), ('sympy', 'sympy'), ('pandas', 'pandas'),
                   ('matplotlib', 'matplotlib')]:
    if importlib.util.find_spec(_mod) is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', _pkg], check=True)
import numpy as np, pandas as pd, scipy
print('numpy', np.__version__, '| scipy', scipy.__version__, '| pandas', pd.__version__)'''

PLOTSTYLE = r'''# ==== 図のスタイル（明るい背景・細い線・識別は色＋線種＋マーカー）====
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'text.color': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.grid': True,
                     'grid.color': '#e4e3df', 'grid.linewidth': 0.8, 'lines.linewidth': 2,
                     'lines.markersize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 10, 'legend.frameon': False})
C_KOB, C_LIT, C_AUX = '#2a78d6', '#eb6834', '#1baf7a'     # categorical slots 1,2,3 (validated palette)
C_R, C_G, C_B = '#e34948', '#008300', '#2a78d6'            # colours of the r, g, b qubit species'''


def md(s):
    return new_markdown_cell(s.strip('\n'))


def code(s):
    return new_code_cell(s.strip('\n'))


def lib_cells(path, title_md):
    src = open(path).read()
    cells = [md(title_md)]
    parts = re.split(r'\n(?=# -{20,})', src)
    for p in parts:
        p = p.strip('\n')
        if not p:
            continue
        m = re.match(r'# -{20,}\s*(.*)', p)
        if m:
            cells.append(md(f'### {m.group(1).strip()}'))
        cells.append(code(p))
    return cells


def save(name, cells):
    nb = new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.metadata['language_info'] = {'name': 'python'}
    nb.metadata['colab'] = {'provenance': []}
    nbformat.write(nb, f'{OUT}/{name}')
    print('wrote', name, len(cells), 'cells')


LIB_TITLES = {
    'common_lib.py': '共通ライブラリ（F2 線形代数・ゲート／ガジェット・スケジューラ・Clifford+T 分解・計数ルール R1–R3）',
    'd4_lib.py': 'D4 ライブラリ（境界付き D4 符号・transversal $U$・code switching・疎状態シミュレータ・資源モデル）',
    'msd_lib.py': '15-to-1 ライブラリ（行列・状態ベクトル回路・Fig. 17c 配置・多パッチ格子手術の明示的構成・資源モデル）',
}


def embedded_lib_cells(*paths):
    """Library code embedded into a notebook (Google Colab runs every notebook in its own runtime, so
    notebooks must not load sibling files).  All notebooks are generated from the same sources, so the
    embedded copies are identical."""
    cells = [md(r'''## 共通ライブラリ（埋め込み・関数定義のみ）
Google Colab ではノートブックごとに別のランタイムで実行されるため、他のファイルを読み込まずに済むよう、共通コードをこのノートブックに埋め込んでいる（各ノートの共通コードは同一のソースから自動生成されており、内容は一致する）。
**このセクションは上から順に実行するだけでよく、編集する必要はない**（Colab では見出しの左の三角で折りたためる）。''')]
    for p in paths:
        cells.append(md(f'### {LIB_TITLES[p]}'))
        src = open(p).read()
        parts = re.split(r'\n(?=# -{20,})', src)
        for part in parts:
            part = part.strip('\n')
            if part:
                cells.append(code(part))
    cells.append(code("print('library loaded')"))
    return cells
