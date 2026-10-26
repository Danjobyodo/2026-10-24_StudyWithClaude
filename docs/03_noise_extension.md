# 将来のノイズシミュレーションへの拡張：必要な変更点

今回の実装は「回路定義」「ゲートコストモデル」「資源計数」「出力データ形式」を分けてあるので、以下の順に拡張できる。

## 1. 現在の構造（再利用できる部分）

| 層 | 実装（埋め込みライブラリ／`tools/*.py` 内の名前） | ノイズ拡張での再利用 |
|---|---|---|
| 符号・安定化子の定義 | `d4_lib`: `D4Lattice`, `CodeData`, `PSOp`；`msd_lib`: `rotated_checks` | そのまま使える |
| 測定ガジェット（物理回路） | `common_lib`: `clifford_x_gadget`, `z_check_gadget`, `pauli_check_gadget`；`d4_lib`: `D4Protocol.round_gadgets` | 誤り挿入位置（各ゲート後）をここに追加 |
| スケジュール | `commuting_classes`, `class_layers` | アイドル誤り（各層で何もしない量子ビット）の位置を与える |
| コストモデル | `D4CostData`, `MSDCostData`, `d4_stages`, `msd_stages`, `summarize` | 距離 $d$ と $\kappa$ を誤り率から決めれば、そのまま資源を出せる |
| 出力 | `results/*_stage_costs.csv`, `*_summary.csv`, `d4_blocks.json` | 列（`p_phys`, `p_L_target`, `d_required`, `kappa`）を追加 |

## 2. Litinski 側
1. **回転表面符号と格子手術の回路レベル雑音**：stim で直接シミュレーション可能（`lattice_surgery_zz_check` を誤り付き・$d$ ラウンドに拡張し、検出器と論理観測量を定義、PyMatching で復号）。
2. **注入誤り**：状態注入の誤り率（Li の方法の解析式または stim シミュレーション）→ 15-to-1 の出力誤り $35p_{\rm inj}^3$ ＋ パッチの論理誤り。
3. **成功確率**：$(1-p)^{15}$ 程度の棄却による平均時間の増加（[Lit] 式(9)）を時間に反映。
4. **多段蒸留**（225-to-1 など）・Fig. 18 の配置は tile 配置を追加転記すれば同じ枠組みで計数できる。

## 3. Kobayashi 側（難しい部分）
1. **非 Clifford 回路の雑音シミュレーション**：D4 のラウンドは CCZ を含むため stim では直接扱えない。候補：
   * 小さい $L$：本研究の疎状態シミュレータ（`SparseState`）に Pauli 誤りを挿入（$L\le2$ 程度）。
   * 大きい $L$：[Tak] の Clifford-stabilizer simulation。**適用には同論文の Definition 2 の条件を D4 符号と code switching の各操作について検証する必要がある**（今回は未検証のため採用していない）。
   * 位相 $F(a)$ の局所構造（V8）を利用した専用シミュレータ：誤りがある場合の位相の変化を局所に追跡できる可能性がある（要検討）。
2. **JIT decoder の実装**：非 Abelian flux（$Z$ 安定化子の違反）を時空間で実時間ペアリングする復号器。[Dav] の記述に従い実装し、閾値と $\kappa_{in}$（必要ラウンド数）を数値的に決める。これにより現在の仮定 A1–A3 を置き換える。
3. **Pauli フレームの扱い**：今回の $Z^rZ^b$ 文字列補正（$m_v$ 由来）は Clifford な回路を通るので Pauli フレームで追跡できるが、$S^g_X$ 測定前の $X$ 型フレームは CZ 修飾を通ると Clifford に変わるため、物理的に補正するか、フレームを Clifford 群で追跡する必要がある。
4. **frozen 量子ビットの除去（A8）**：ノイズありでは $\mathcal L_{rb}$ の $V$ の CNOT 対の省略が誤り伝播を変えるので、除去しない版（`D4Protocol(L, prune_frozen=False)`）も比較する。
5. **FSC → 回転表面符号の変換（U1）**：具体的な回路を構成し、stim で検証してから資源に加える。

## 4. 要求精度ごとの最小資源（次段階のゴール）
1. 物理誤り率 $p$ を与え、各方式の出力論理誤り率 $\varepsilon(d,\kappa,p)$ を上のシミュレーションで推定。
2. 要求誤り率 $\varepsilon_{\rm target}$ を満たす最小の $(d,\kappa)$ を選び、`summarize` で体積を計算。
3. $(p,\varepsilon_{\rm target})$ 平面で体積比 $V_{\rm Kobayashi}/V_{\rm Litinski}$ の等高線（優位領域図）を描く。
4. Magic state cultivation との比較も同じ計数ルール（R1–R3）で追加できる。

## 5. 注意
* ノイズありの比較では「ラウンド数の定数」「非 Clifford ゲート数」「変換コスト」が支配的になりうるため、今回の理想条件下の比（体積で約 0.1〜0.2）はそのまま外挿できない。
