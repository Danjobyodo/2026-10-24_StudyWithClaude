# 設計方針：D4符号のcode switching（Kobayashi方式）と15-to-1蒸留（Litinski方式）の理想条件下資源比較

本書は実装着手前に整理した比較単位・モデル化の仮定・必要な回路・ファイル構成・正当化方法をまとめたものである。
実装中に判明した事実（境界付きD4符号の距離など）は「実装後の確定事項」として追記してある。
仮定・未解決事項の一覧は [`02_assumptions_and_open_issues.md`](02_assumptions_and_open_issues.md) を参照。

---

## 0. 文献の位置づけ

| 略号 | 文献 | 本研究での役割 |
|---|---|---|
| [KZH] | Kobayashi, Zhu, Hsin, *Clifford Hierarchy Stabilizer Codes: Transversal Non-Clifford Gates and Magic*（arXiv:2511.02900v3、本文＋SM） | D4符号の安定化子（式(2),(3)、Fig. 4）、境界条件（式(7)）、automorphism $U=WV$（式(5),(6),(8)）、code switchingプロトコル（本文 p.4、SM Sec. V A） |
| [Lit] | Litinski, *A Game of Surface Codes*（Quantum 3, 128 (2019)） | 15-to-1回路（Fig. 14, 15, 式(1)）、auto-corrected π/8回転（Fig. 17a,b）、11 tiles × 11τ配置（Fig. 17c）、tile↔物理量子ビット・cycle対応（Sec. 1, App. A, Sec. 4.3）、状態注入（App. A, Fig. 38） |
| [Tak] | Takada, Bartlett, Williamson, *Exact efficient simulation of noisy logical magic states using Clifford stabilizers* | **参考資料のみ。採用しない。**（理由：今回はノイズなしで、必要な検証はより単純な厳密手法で完結する。D4符号・code switchingへの適用条件（同論文 Definition 2）は未検証のため前提にしない。） |
| [Dav] | Davydova et al., *Universal fault tolerant quantum computation in 2D without getting tied in knots*（arXiv:2503.15751） | [KZH] が参照するJIT decoderの出典。**本環境からは全文にアクセスできず（arXivへの接続が遮断）、具体的なラウンド数を確認できなかった。** したがって $O(d)$ ラウンドは定数 $\kappa$ でパラメータ化する。 |

---

## 1. 比較単位

### 1.1 共通のゴール
両方式とも出力は surface-code 系の論理状態 $|T_L\rangle = T_L|+_L\rangle$ 1個。

* [KZH] の $U$ は論理 $T^\dagger$ を与え、出力は $T^\dagger|+\rangle$（folded surface code, FSC 上）。
* [Lit] の15-to-1も出力は $T^\dagger|+\rangle$（[Lit] Sec. 3.1 に "actually $T^\dagger$" と明記。本研究で状態ベクトル計算により確認）。
* 共通化：$X\,T^\dagger|+\rangle = e^{-i\pi/4}\,T|+\rangle$ なので、両方式とも論理 $\bar X$（Pauli）をPauliフレームで適用して $T|+\rangle$ に統一する。**コスト0（両方式で同一の扱い）。**
* [KZH] の出力は FSC（2層の $\mathbb Z_2$ 表面符号を折り畳んだ三角形パッチ）であり、[Lit] の通常の回転表面符号パッチと直接接続できるとは仮定しない。FSC→通常パッチ変換は **未解決（比較Bで除外、または「例示的仮定」として別表示）**。

### 1.2 距離 $d$ の合わせ方
* [Lit]：距離 $d$ の回転表面符号パッチ。
* [KZH]：$L\times L$ 正方格子（Fig. 4 は $L=9$）。**FSCの符号距離が $d$ となる最小の $L$** を採用する。厳密MILPで $d_Z = L$、$X$型シフト距離 $= L+3$（$L=1,\dots,10$）を得たので $L=d$。

### 1.3 計数ルール（両方式で共通）
* **R1 物理量子ビット**：データ量子ビット＋測定する各安定化子生成元に1個の測定補助量子ビット。単一量子ビット $Z$ 安定化子で固定される境界量子ビット（frozen qubit）は割り当てない。データのみの値も併記。
* **R2 時間**：(a) 安定化子測定ラウンド数（code cycle）を符号ごとに数える。(b) 物理時間 = 層の所要時間の和。下界（最も忙しい量子ビットのゲート数）と上界（妥当性を保証した可換クラス・スケジュール）の両方を出す。**異なる符号のラウンドは同一視しない**（D4のラウンドはCCZを含み深い）。
* **R3 時空間体積**：各段階の使用量子ビット数×その段階の時間 を積算（qubit·round と qubit·time-unit）。ピーク×総時間は「上界」と明記して別記。
* ゲートセット：既定は {INIT, MEAS, H, CNOT, CZ, CCZ, CS, T} を原始ゲート（所要時間はパラメータ、既定は全て1単位）。CCZ・CSを Clifford+T に分解したモデル（CCZ: T 7個 + CNOT 6個、CS: T 3個 + CNOT 2個、分解は検証ノートで行列として確認）も併記。分解でアンシラや追加の非Clifford資源は要らない構成を用いる。

### 1.4 時間モデル（両方式に同じ考え方で適用）
| モデル | [KZH] | [Lit] |
|---|---|---|
| `FT_ref`（耐故障参照） | S0: $\lceil\kappa_0 d\rceil$、S1: $\lceil\kappa_{in} d\rceil$、S4: $\lceil\kappa_{out} d\rceil$ ラウンド（既定 $\kappa=1$、**仮定**） | 1τ = $d$ ラウンド（**文献**）|
| `NL_min`（ノイズなし最小） | S0: 1ラウンド、S1: $S^g_X$のみ1ラウンド、S4: 0 | 1τ = 1ラウンド |

`NL_min` は「ノイズなしなら繰り返し測定は不要」という論理的最小であり、耐故障性の主張ではない。

### 1.5 比較A／比較B
* **比較A（プロトコル本体）**：[Lit] は入力 $|m\rangle$ が供給済み（注入時間0、物理Tゲート数に含めない）。[KZH] は S0〜S4（FSC準備、switch-in、$U$、switch-out、EC）。
* **比較B（可能な範囲での完全比較）**：[Lit] に状態注入（[Lit] App. A の記述：3+3 cycle、$d$に依存しない）と物理Tゲート15個、11-tile配置での注入の直列化（[Lit] "Caveat"）を加える。[KZH] は CCZ/CS/T の物理非Clifford資源をそのまま計上し、Clifford+T分解時のT数も示す。FSC→通常パッチ変換は未解決のため除外（下界）＋例示的仮定（1τの格子手術）を別表示。

---

## 2. 必要な回路と実装

### 2.1 D4符号（[KZH]）
* 格子：頂点 $(x,y)$、$0\le x,y\le L$、辺は $+x$, $+y$ 向き（branching）。plaquette の頂点ラベル 0=(x,y),1=(x+1,y),2=(x,y+1),3=(x+1,y+1)（Fig. 1）。
* カップ積：SM Sec. I の超立方体公式 $(a\cup b)(p)=a(e_{01})b(e_{13})+a(e_{02})b(e_{23})$。
* 安定化子：式(2)(3)の $S^{c}_{X;v}$、$S^c_{Z;f}$。境界は Fig. 4 / 式(7)：左 $\mathcal L_r$（$Z^r$）、上 $\mathcal L_b$（$Z^b$）、下・右 $\mathcal L_{rb}$（$Z^rZ^b$, $Z^g$）。境界頂点で許される $X$ 型生成元は「$X$ 部分が境界 $Z$ と可換な色の組」として自動導出（$\mathcal L_{rb}$ では $S^r_XS^b_X$、Fig. 4 と一致）。範囲外の plaquette の CZ は落とす（Fig. 4 の描画と一致）。**トーラスの周期境界は使わない。**
* 表現：$X$型Clifford安定化子は $O|x\rangle=(-1)^{f(x)}|x+s\rangle$（$f$ は2次）で厳密に表せる。非可換性は「$Z$安定化子部分空間 $C_Z$ 上での可換性」として $\mathbb F_2$ 線形代数で厳密判定（勝手に無視しない）。
* $U=WV$：$V=\prod_e \mathrm{CNOT}^{(r,g)}\mathrm{CNOT}^{(b,g)}$、$W=\prod_p \mathrm{CS}^{r,b}_{e_{01},e_{13}}(\mathrm{CS}^{r,b}_{e_{02},e_{23}})^\dagger\prod_{e\in\mathcal L_{rb}}(T^r_e)^\dagger$（式(8)）。

### 2.2 Code switching（[KZH] 本文 p.4・SM Sec. V A）
S0 FSC $|+_L\rangle$ と $|0\rangle^g$ → S1 $S^g_X$（とその他全安定化子）の測定（結果 $m_v$ はランダム）→ S2 $U$ → S3 $Z^g_e$ 測定と $\prod_{v\in\kappa}S^g_{X;v}$ による補正 → S4 FSC のEC。
ノイズなしで必要な処理（本研究で導出・検証）：
* $m_v=1$ の $e_g$ 電荷は $U$ によって $e_{rgb}$ に写る。復帰後は FSC の $A^r_v,A^b_v$ が $-1$ になるので、$\mathcal L_{rb}$ から $v$ への $Z^rZ^b$ 文字列（Pauli）で補正（Pauliフレームで物理コスト0）。
* $Z^g$ 測定結果は $\delta\kappa$（完全形）になり、$\prod_{v\in\kappa}S^g_{X;v}$ のCZ部分（Clifford）を物理的に適用する（[KZH] の記述どおり）。
* JIT decoder・matching/RG decoder は**ノイズなしでは非自明な復号処理が不要**（シンドロームは決定的またはプロトコル由来のランダム値のみ）。その古典計算コストは計上しない。

### 2.3 15-to-1（[Lit] Sec. 3）
* 論理レベル：式(1)の行列 $M$ から Fig. 15 の5量子ビット回路（11回転）を生成。Fig. 14 の15量子ビット符号版とも照合。
* auto-corrected π/8 回転（Fig. 17b）を全測定分岐で検証。
* tileモデル：Fig. 17c の配置を格子データとして記述し、使用tile数・時間ステップ数を再計数。
* 物理モデル：回転表面符号パッチの実際の量子ビット数（$2d^2-1$）、2-tileパッチ、格子手術のアンシラ領域（1 tile 当たりの増分をstimで検証した値）から段階ごとの量子ビット数を算出。ラウンドの物理時間は D4 と同じスケジューラで計算。

---

## 3. ファイル構成

| ファイル | 役割 |
|---|---|
| `common_lib.ipynb` | $\mathbb F_2$ 線形代数、ゲート／ガジェット定義、スケジューラ、Clifford+T分解、共通コストモデル（計数ルール R1〜R3） |
| `d4_lib.ipynb` | D4格子・安定化子・符号データ・距離（MILP）・$U$・プロトコル位相・疎状態シミュレータ・D4資源モデル |
| `msd_lib.ipynb` | 15-to-1 の行列・状態ベクトル回路・tile配置・回転表面符号パッチと格子手術 |
| `D4_switching_main.ipynb` | D4側の資源計数（段階別）、結果を `results/` に保存 |
| `D4_switching_validation.ipynb` | D4側の正当性検証（独立実装を含む） |
| `MSD_15to1_main.ipynb` | Litinski側の資源計数（tileモデル・物理モデル） |
| `MSD_15to1_validation.ipynb` | Litinski側の正当性検証 |
| `resource_comparison.ipynb` | 比較表・グラフ・コスト比、未確定事項の一覧表示 |

ライブラリノートブックは関数定義のみで重い計算をしない。他のノートブックは `%run ./common_lib.ipynb` 等で読み込む（Google Colab ではすべての `.ipynb` を同じフォルダに置く）。

出力：`results/*.csv, *.json`、`figures/*.png`。乱数はシード固定。

---

## 4. 正当化の方法（主結果と独立に検証）

1. **符号の数学的正当性**：生成元の $C_Z$ 保存、$C_Z$ 上の対ごとの可換性、$S^2=1$、論理次元 $k=1$、論理演算子、距離（MILP）。$L=1$ では行列による完全に独立な実装と照合。$L=2$ では Knill–Laflamme 条件を直接検査。
2. **$U$ の正当性**：各生成元について $US U^\dagger = S'$ を局所厳密列挙（$C_Z$ の局所射影上で全通り）で確認し、論理位相を厳密計算。符号規約を変えた $W$ では失敗すること（負の対照）も示す。
3. **プロトコル全体**：$L=1,2$ で疎状態ベクトルによる厳密シミュレーション（ランダム入力・全測定分岐）。より大きな $L$ では、各配置が受け取る位相 $F(a)$ を導出し、それが論理剰余類上で一定であることを局所厳密列挙で証明（$L\le 8$）。補正を外すと失敗すること（負の対照）も示す。
4. **測定回路**：各ガジェット（制御 $S$ ＋ $X$ 測定）が理想射影測定と一致することを密行列で検証。可換クラス内のゲートが互いに可換であることを独立に（小行列で）検証。
5. **15-to-1**：三直交性、$M$ と $\tilde G$ の整合、理想入力→理想出力、1・2個の誤りの検出、3個誤りの未検出数35、auto-corrected π/8 の全分岐、tile／time step の再計数、物理側の格子手術（stim）。
6. **資源計数**：段階別の数値が回路から直接数えたものであることを、独立な数え方（閉形式多項式）と照合。

**検証していない事項は「未検証」と表示する**（例：FSC→通常パッチ変換、$O(d)$ の定数、JIT decoderの性能。D4 相の Knill–Laflamme 距離は改訂で $L\le5$（6）まで証明、それより大きい $L$ は未証明）。

> 改訂（外部レビュー対応）：Litinski の多パッチ測定の物理レベル構成（M9）、D4 相の KL 距離の厳密下界（V5b）、回路と位相公式の照合（V7b）、比較Bの未計上コストの分離、感度分析を追加した。詳細は [`04_review_response.md`](04_review_response.md)。
