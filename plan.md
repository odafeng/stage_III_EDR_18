# 研究計劃（完整稿 v1.0）

> 專案代號：EDR-III-Colon-ML
> 版本：v1.0
> 日期：2025-11-02
> 設計類型：回溯性、雙中心（衍生＋外部驗證），TRIPOD Type 3；若外驗資料延後，A 院先行 Type 2b（時間切割）內部驗證

---

## 題名（暫定）

**中文**：以常規術後病理與臨床變項建立機器學習模型預測 Stage III 結腸癌術後 18–24 個月內遠端復發之風險：與 AJCC 分期與淋巴結比例之比較，並於第二中心進行外部驗證之雙中心回溯性研究

**英文**：Machine-learning prediction of early distant recurrence within 18–24 months after curative surgery for stage III colon cancer using routine clinicopathologic variables: comparison against AJCC stage and lymph node ratio with external validation at a second center (two-center retrospective study)

---

## 摘要（150–200 字）

本研究擬以**術後常規**臨床與病理變項建立預測 **早期遠端復發（Early Distant Recurrence, EDR）** 的機器學習模型（主視窗 18 個月、敏感度視窗 24 個月），並**與 AJCC 分期與淋巴結比例（LNR）**比較效能。設計為**雙中心回溯性**研究：A 院進行模型衍生與內部驗證，模型凍結後於 B 院進行**外部驗證**（TRIPOD Type 3）。以 ROC-AUC、PR-AUC、校準（calibration-in-the-large、slope、Brier）與 Decision Curve 評估，並以 SHAP 解析關鍵風險因子與提出 ML 衍生高風險表現型。期望證實僅憑常規病理即可辨識術後 18–24 個月內遠端轉移高風險族群，具有臨床可行性與跨中心可遷移性。

---

## 背景與臨床動機

* Stage III 結腸癌異質性高；部分患者術後 1–2 年內即發生遠端轉移，預後顯著較差。
* 傳統 **AJCC 8th** 與 **LNR** 能力有限，難同時整合 **LVI、PNI、tumor deposits、mucinous、MSI、營養（albumin）** 等常規但分散的訊息。
* **聚焦 EDR**（而非所有復發）能鎖定生物學較侵襲的族群、降低異質性與長期管理差異的干擾，對**術後前兩年**的影像追蹤與試驗轉介最具行動性。
* 以 ML 整合多變項與非線性關係，於常規資料框架內提供更精準的風險分層與可解釋性（SHAP）。

---

## 研究問題與假設

**主要研究問題**：常規術後病理＋臨床變項之 ML 模型，能否優於 AJCC 與 LNR，準確預測 Stage III 結腸癌術後 **18 個月內遠端復發**？

**主要假設**：

1. ML 模型（XGBoost 為主）之 ROC-AUC 與 PR-AUC **高於** AJCC-only 與 LNR-only 基準。
2. 模型於 B 院外部驗證具可接受之**區分度**與**校準度**；必要時可經**重標定**（調整截距/斜率）改善校準。
3. SHAP 能重現一組穩定之**高風險表現型**（例如 LVI/PNI/deposits、較高 LNR、mucinous、CEA 偏高、albumin 偏低等組合）。

---

## 研究設計

* 類型：回溯性、雙中心（A：衍生＋內部驗證；B：獨立外部驗證）。
* 報告準則：**TRIPOD+AI**；品質自評：**PROBAST**。
* 若外部資料未即時到位，A 院先採 **時間切割（Type 2b）** 完成內部驗證。

---

## 研究對象

### 納入條件

1. 病理確診 **Stage III 結腸癌**（AJCC 8th：IIIA/IIIB/IIIC）。
2. 完成根除性手術，具備復發與存活追蹤資訊。

### 排除條件

1. 術後 90 天內死亡或隨訪不足視窗長度（18/24 個月）。
2. 無法判定復發型態或日期。
3. 非結腸來源（直腸癌等）。
4. 術前/同期遠端轉移。

---

## 終點定義

* **主要終點：EDR-18m** ＝ `Recurrence==1` 且 `Recurrence_Type==Distant` 且 `Recurrence_Date − Radical_Op_Date ≤ 18 個月`；個案需具備 ≥18 個月可判定之隨訪。
* **敏感度終點：EDR-24m**（視窗改 24 個月）。
* **探索性終點**：Any-recurrence（含局部）、Time-to-any-recurrence（DFS 視角）、OS（若可）。

---

## 候選特徵（依實際欄位對應）

* **臨床/一般**：Age、Sex、BMI、ECOG、PreOp_Albumin、Dx_Year。
* **腫瘤位置**：Tumor_Location、Tumor_Location_Group（右側/左側/橫結腸等）。
* **分期/病理**：pT、pN、AJCC_Substage、Tumor_Size_cm、Differentiation、Histology、**LVI**、**PNI**、**Tumor_deposits**、Mucinous_any / >50%、Signet_ring、**MSI_status**。
* **淋巴結**：LN_total、LN_positive、**LNR = LN_positive/LN_total**。
* **血清**：CEA_preop（及 log 轉換）。
* **手術/醫師**：Op_Procedure、Visiting_Staff（practice proxy）。

> 以資料字典對齊兩中心的欄位名稱/單位/編碼（LVI、PNI、deposits、mucinous、MSI 定義一致）。

---

## 資料處理

1. **缺值**：連續以 training 中位數補值；類別以眾數或新增「Unknown」；所有轉換僅以 **訓練集統計量** 擬合再套用至驗證/外驗（避免資料洩漏）。
2. **縮放/變換**：視需要對 CEA 取對數；可對連續變項做 Winsorization（1–99%）。
3. **編碼**：類別以 one-hot；極小類別合併。
4. **不平衡**：`class_weight=balanced` 或 SMOTE（僅訓練集）。
5. **切割**：A 院以年度做 **temporal split**（如 2017–2021 訓、2022–2024 驗）；或 80/20 stratified + 5-fold CV。B 院作**完全獨立外部驗證**，**不得在 B 院調參**。

---

## 基準與比較策略

### 傳統基準

* **AJCC-only**：

  * 有序分數：IIIA=1、IIIB=2、IIIC=3 作為風險分數計算 ROC/PR-AUC；
  * 經驗機率：以訓練集分期別的 EDR 實際比例作為預測機率；
  * 敏感度：以 AJCC 為唯一自變項建立 logistic，輸出機率。
* **LNR-only**：以連續 LNR 作為風險分數；並以文獻切點分層作敏感度分析。

### 機器學習模型（控制在 2–3 種）

* **Logistic Regression（L1）**：臨床可解釋基準。
* **Random Forest**：非線性、對不平衡較穩。
* **XGBoost（主力）**：5-fold CV 調參；用 SHAP 解釋。

> 交叉驗證與超參數搜尋僅於訓練資料內完成；最終**凍結模型**（含前處理、特徵清單、超參數與決策閾值）。

---

## 外部驗證設計（B 院）

1. **對齊**：欄位映射、單位一致、病理定義同步；保證事件定義與視窗一致。
2. **評估**：ROC/PR-AUC、Calibration-in-the-large、Calibration slope、校準曲線、Brier、Decision Curve；主要亞族群（右/左側、MSI、≥75 歲、LVI/PNI/deposits）。
3. **重標定規劃**：若校準偏離，優先進行**截距/斜率校準**（不改模型結構與特徵）。
4. **資安/治理**：去識別化、DUA/IRB；如資料不能流出，採本地執行、回傳指標與圖表。

---

## 評估指標與統計

* **區分度**：ROC-AUC、**PR-AUC**（不平衡重點）；DeLong/bootstrapping 取得 95% CI。
* **校準**：Calibration intercept、slope、Brier、校準曲線。
* **臨床效益**：Decision Curve（機率閾值 5–30%）；報告 net benefit。
* **Operating point**：固定 80% specificity 報 sensitivity/PPV/NPV；另報 Youden index 閾值。
* **統計軟體**：Python（pandas、scikit-learn、xgboost、lifelines/scikit-survival）、或 R（tidyverse、xgboost、rms、rmda）。

---

## 敏感度與探索性分析

1. 視窗：EDR-24m 取代 18m。
2. 終點：Any-recurrence、Time-to-any-recurrence（Cox / Random Survival Forest）。
3. 人群：排除 MSI-H、或分層（MSI-H vs non–MSI-H）。
4. 指標：以 LNR 切點分層 vs 連續值；以 Albumin cut-off vs 連續值。
5. 取樣：類別權重 vs SMOTE；不同補值策略（中位數 vs KNN）影響。

---

## 樣本量與事件數考量

* 以**事件數**為主（EDR 事件）。若事件有限，採**正則化（L1）**、控制模型複雜度；報告 95% CI 與校準指標以反映不確定性。
* 外部驗證理想上 ≥100 事件；不足時務必呈現 PR-AUC 與校準曲線，避免僅報 ROC-AUC。

---

## 可解釋性與模型卡

* 以 **SHAP** 呈現全體與個案層級的重要特徵與方向性；
* 產出 **模型卡（Model Card）**：資料期間、納入/排除、特徵清單、前處理步驟、超參數、訓練/內驗/外驗效能、建議閾值、適用族群與限制。

---

## 倫理、資料治理與可重現性

* IRB 類別：回溯性資料庫研究；去識別化處理；僅報整體統計。
* DUA：跨中心傳輸規範與資安；必要時採本地運算模式。
* 可重現性：版本控管（Git 標記）、鎖定亂數種子、紀錄軟體版本；附分析腳本與模型卡於補充資料。

---

## 報告準則（TRIPOD+AI）對照摘要

* 明確描述資料來源、納排準則、預測目標、特徵、樣本大小與事件數。
* 詳述缺值處理、特徵工程、切割策略、超參數搜尋、模型凍結。
* 提供區分度、校準、臨床效益、置信區間；附圖（ROC/PR、校準、DCA）。
* 外部驗證與重標定流程、決策閾值、使用情境與限制。

---

## 風險與緩解

* **事件數不足**：合併 24m 視窗敏感度；報 PR-AUC、寬 CI；用正則化與簡化模型。
* **跨中心異質**：嚴格對齊定義；必要時僅作重標定；在敏感度中加入 center effect（固定或分層）。
* **監測偏差**：固定視窗終點；探索性用 IPCW 或時間到事件模型佐證。

---

## 里程碑與時程（參考）

1. 週 1–2：IRB/DUA、資料字典對齊、抽取與去識別。
2. 週 3–4：資料清理、缺值/編碼、EDR 標籤建立；Table 1（A 院）。
3. 週 5–6：A 院模型衍生（CV 調參、內驗）、SHAP、與 AJCC/LNR 比較。
4. 週 7–8：模型凍結＋文件；B 院外部驗證（指標與圖表回傳）。
5. 週 9：重標定（若需）、完成 DCA 與亞族群分析。
6. 週 10：撰稿、圖表與補充文件、TRIPOD+AI 檢核。

---

## 預期產出

* 論文一篇（目標：International Journal of Colorectal Disease / BMC Cancer / Cancers / Annals of Coloproctology）。
* 圖表：流程圖、ROC/PR、校準、DCA、SHAP、亞族群森林圖、兩中心基線表。
* 附件：模型卡、程式碼（或偽代碼）、TRIPOD+AI/PROBAST 檢核表。

---

## 作者與分工（ICMJE 指引）

* 概念與設計：A（PI）、B（共同通訊）。
* 資料收集/整理：A 中心團隊、B 中心團隊。
* 分析與解釋：資料科學與臨床共同參與。
* 撰稿與審閱：全體作者。
* 經費/資源：若有，註明來源；否則載明無特定經費支持。

---

## 附錄 A：AJCC 與 LNR 的 AUC 實作（概念偽代碼）

```python
# y_true: EDR_18m (0/1)
# AJCC ordinal score
ajcc_map = {"IIIA":1, "IIIB":2, "IIIC":3}
y_score_ajcc = df["AJCC_Substage"].map(ajcc_map)
roc_auc_score(y_true, y_score_ajcc)

# AJCC empirical probability（以訓練集內的實際比例取代）
p_stage = df_train.groupby("AJCC_Substage")["EDR_18m"].mean()
y_prob_ajcc_emp = df_test["AJCC_Substage"].map(p_stage)
roc_auc_score(y_true_test, y_prob_ajcc_emp)

# LNR-only（連續值）
roc_auc_score(y_true, df["LNR"])  # 或以分層 cutoffs 當分數
```

---

## 附錄 B：EDR 標籤建立（概念偽代碼）

```python
edr_18m = (
    (df["Recurrence"].eq(1)) &
    (df["Recurrence_Type"].str.lower().eq("distant")) &
    ((df["Recurrence_Date"] - df["Radical_Op_Date"]).dt.days <= 18*30.44)
)
```

---

## 圖表清單（建議）

1. 研究流程圖（納入/排除、A/B 院樣本與事件數）。
2. 兩中心基線表（含缺值概況、關鍵變項分布）。
3. ROC 與 PR 曲線（A 院內驗、B 院外驗）。
4. 校準曲線（外驗前/重標定後）。
5. Decision curve（A 與 B）。
6. SHAP summary plot 與個案 force plot。
7. 亞族群森林圖（右/左側、MSI、≥75 歲、LVI/PNI/deposits 等）。

---

## 稿件結構（提要）

1. **Introduction**：問題、缺口、EDR 合理性、研究目的。
2. **Methods**：設計、人群、終點、特徵、前處理、模型與比較、評估指標、外部驗證、統計方法、倫理。
3. **Results**：流程、基線、模型效能（AUC/PR/校準/DCA）、SHAP、敏感度與亞族群、外部驗證（含重標定）。
4. **Discussion**：主要發現、臨床意義、與文獻比較、限制、未來工作（多中心/前瞻驗證/臨床實裝）。
5. **Conclusion**：常規病理驅動的 ML 能在術後早期提供更佳風險分層並具可遷移性。
