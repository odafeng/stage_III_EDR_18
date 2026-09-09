# Verification Report — Verifier-2（影片數字 vs Manuscript_final.docx）

## 摘要

- 驗證員：Verifier-2（與 Verifier-1 完全獨立，未參考其產出）
- 影片來源：`video_build/va/va_deck.html`（畫面文字 + data-cap caption）、`video_build/va/scenes.json`（旁白 text）
- 權威來源：`Submission/Manuscript_final.docx`（輔助：`Submission/Response_to_Reviewers.docx`）
- 共抽出 **39 條**影片數字/數據宣稱（含 caption 與旁白）
- 結果：**MATCH 39 條／MISMATCH 0 條／UNVERIFIABLE 0 條**
- 逐條比對僅以 Manuscript_final.docx 為準；Manuscript 自身的內部不一致處另列於「Manuscript 內部觀察」

## 逐條比對表

| # | 場景/來源 | 影片宣稱 | Manuscript 數值（原文引用） | 狀態 | 備註 |
|---|-----------|----------|------------------------------|------|------|
| 1 | S1 標題 (data-cap / h1+h3) | 「a simple 4-variable ML model with external validation」；Four-Variable Machine Learning Model | Title: “Ruling Out Early Distant Recurrence in Stage III Colon Cancer: A Simple Four-Variable Machine Learning Model with External Validation” | MATCH |  |
| 2 | S1 標題 (旁白 s1) | 「a simple four-variable machine learning model」 | 同上（four-variable model） | MATCH | 旁白口語化，與標題一致 |
| 3 | S2 問題 (h2 + data-cap) | 「20–30% of stage III colon cancer patients recur」；「20–30% of stage III patients recur after curative surgery」 | Introduction: “Despite curative resection, outcomes vary widely,4 with 20% to 30% of patients experiencing recurrence.” | MATCH |  |
| 4 | S2 問題 (旁白 s2) | 「twenty to thirty percent of stage three patients recur」 | 同上（20% to 30%） | MATCH |  |
| 5 | S2 問題 (stat + data-cap) | 「<18 mo Early distant recurrence (EDR)」；data-cap「≤18 mo」 | Abstract: “Early distant recurrence within 18 months reflects aggressive tumor biology…”; Methods: “EDR-18, defined as radiologically or pathologically confirmed distant metastasis occurring within 18 months after curative surgery.” | MATCH | 注意：影片自身在「<18 mo」(畫面上) 與「≤18 mo」(caption) 之間措辭不一致；Manuscript 用「within 18 months」(≈≤18)。數值含義一致，無數字衝突。 |
| 6 | S2 問題 (旁白 s2) | 「Early distant recurrence within eighteen months signals aggressive tumor biology」 | Abstract: “Early distant recurrence within 18 months reflects aggressive tumor biology in stage III colon cancer.” | MATCH |  |
| 7 | S2 問題 (stat) | 「Aggressive / Tumor biology signal」 | 同上（aggressive tumor biology） | MATCH | 定性宣稱 |
| 8 | S3 缺口 (card) | 「Wide outcome variation within the same substage — especially stage IIIB」 | Introduction: “…AJCC stage IIIB, a subgroup known for its biological heterogeneity.”; Discussion: “AJCC stage IIIB encompasses a broad biological spectrum… substantial heterogeneity even within the same substage.” | MATCH | 定性宣稱 |
| 9 | S3 缺口 (card) | Molecular tools (ctDNA)「costly, not universally available」 | Introduction: “Liquid biopsy techniques like ctDNA offer promise but are limited by cost and access barriers.” | MATCH | 定性宣稱 |
| 10 | S4 模型 (h2 + cards) | 「four routine variables」：AJCC substage (IIIA/IIIB/IIIC)、lymph node ratio、perineural invasion、tumor differentiation | Methods (Feature selection): “…resulted in four predictors: AJCC substage, lymph node ratio, perineural invasion, and tumor differentiation.”; substages IIIA/IIIB/IIIC 出現於 Methods | MATCH |  |
| 11 | S4 模型 (stat) | 「331 / Derivation cohort · 62 events」 | Abstract: “331 patients in the derivation cohort (62 events)”; Results: “The derivation cohort included 331 patients with stage III colon adenocarcinoma”; Methods: “With 62 EDR-18 events…” | MATCH |  |
| 12 | S4 模型 (旁白 s4) | 「three hundred thirty-one patients」 | 同上（331） | MATCH |  |
| 13 | S4 模型 (stat) | 「142 / External cohort · 19 events」 | Abstract: “142 in the external cohort (19 events)”; Methods: “…142 patients undergoing surgery between January 2017 and December 2021, including 19 EDR-18 events” | MATCH |  |
| 14 | S4 模型 (旁白 s4) | 「independent cohort of one hundred forty-two」 | 同上（142） | MATCH |  |
| 15 | S4 模型 (stat) | 「TRIPOD+AI / Reporting guidance」 | Methods: “The study followed the TRIPOD+AI recommendations for transparent reporting of prediction model development and validation.” | MATCH |  |
| 16 | S5 內部 (stat + data-cap) | 「0.680 / AUC (AJCC: 0.625)」 | Results: “The four-variable XGBoost model achieved a pooled out-of-fold AUC of 0.680 (95% CI 0.608–0.751)”; “…AJCC substage alone achieved a pooled out-of-fold AUC of 0.625 (95% CI 0.545–0.698)”; Figure 2 legend: “0.680 vs. 0.625” | MATCH |  |
| 17 | S5 內部 (stat + data-cap) | 「77.4% / Sensitivity」 | Results: “…the ML model demonstrated higher sensitivity (77.4% vs. 45.2%)…”; Abstract: “Sensitivity (77.4%)…” | MATCH |  |
| 18 | S5 內部 (stat + data-cap) | 「89.8% / Negative predictive value」 | Results: “…negative predictive value (89.8% vs. 87.1%)…”; Abstract: “negative predictive value (89.8%)…” | MATCH |  |
| 19 | S5 內部 (旁白 s5) | 「AUC of zero point six eight」= 0.68 | Manuscript 為 0.680（等值，四捨五入至 2 位） | MATCH | 旁白取 2 位小數，與 0.680 一致 |
| 20 | S5 內部 (旁白 s5) | 「seventy-seven percent sensitivity」= 77% | Manuscript 為 77.4%（四捨五入至整數 = 77%） | MATCH | 旁白取整數，與 77.4% 一致 |
| 21 | S5 內部 (旁白 s5) | 「ninety percent negative predictive value」= 90% | Manuscript 為 89.8%（derivation）；Discussion 亦以「approximately 90%」概括兩 cohort | MATCH | 旁白取整數，與 89.8% 及 manuscript「~90%」框架一致 |
| 22 | S5 內部 (旁白 s5) | 「higher than AJCC alone」 | Results: sensitivity 77.4% vs 45.2%、NPV 89.8% vs 87.1%、AUC 0.680 vs 0.625（數值均較高；AUC 差異未達統計顯著，DeLong p = 0.062，影片未宣稱顯著性） | MATCH | 影片未聲稱 AUC 差異具統計顯著性 |
| 23 | S6 外部 (stat + data-cap) | 「0.633 / External AUC」 | Results: “…the four-variable model achieved an AUC of 0.633…”; Discussion: “with an AUC of 0.633 (95% CI 0.501–0.757)”; Figure S3: “AUC = 0.633” | MATCH |  |
| 24 | S6 外部 (stat + data-cap) | 「90.8% / Negative predictive value」 | Abstract: “…a negative predictive value of 90.8%…”; Discussion: “an NPV of 90.8% (95% CI 83.0–97.1%)” | MATCH |  |
| 25 | S6 外部 (stat + data-cap) | 「HR 1.83 / DFS separation (p = 0.030)」 | Results: “…(HR 1.83, 95% CI 1.06–3.16; p = 0.030; Supplementary Figure S4).” | MATCH | 見 Manuscript 內部觀察 #1（Fig S4 圖註另列 log-rank p = 0.028） |
| 26 | S6 外部 (data-cap) | 「sensitivity 68.4%」 | Discussion: “…including an external sensitivity of 68.4%…” | MATCH | 68.4% 僅出現在影片 caption（非畫面上 stats），Manuscript 於 Discussion 支持 |
| 27 | S6 外部 (旁白 s6) | 「AUC zero point six three」= 0.63 | Manuscript 為 0.633（四捨五入至 2 位 = 0.63） | MATCH | 旁白取 2 位小數，與 0.633 一致 |
| 28 | S6 外部 (旁白 s6) | 「ninety point eight percent negative predictive value」= 90.8% | 同上（90.8%） | MATCH |  |
| 29 | S6 外部 (旁白 s6) | 「clear separation in disease-free survival between risk groups」 | Results: “ML-defined risk groups demonstrated significantly different DFS outcomes… (HR 1.83 … p = 0.030)” | MATCH |  |
| 30 | S7 SHAP (sub + data-cap + 旁白 s7) | 「AJCC IIIC · lymph node ratio · perineural invasion — the strongest risk drivers」；旁白「stage three C, lymph node ratio, and perineural invasion drove risk」 | Results: “SHAP analysis identified AJCC substage IIIC as the most influential predictor of EDR-18, followed by lymph node ratio and perineural invasion (Figure 4).” | MATCH | 前三名驅動因子與順序一致 |
| 31 | S8 臨床 (flow + data-cap) | 「4 pathology inputs」 | Methods (Model Deployment): “The tool accepts four routine inputs and outputs both predicted EDR-18 probability and a high- or low-risk classification.” | MATCH |  |
| 32 | S8 臨床 (foot-link) | 「stage3edr.streamlit.app」 | Methods: “…the online tool at https://stage3edr.streamlit.app.” | MATCH |  |
| 33 | S9 限制 (card + data-cap) | 「Both cohorts from two Taiwanese institutions」 | Methods (Fairness): “Both cohorts were drawn from two Taiwanese institutions…” | MATCH |  |
| 34 | S9 限制 (data-cap + 旁白 s9) | 「limited external events」 | Limitations: “…small numbers of events in the external cohort.”; “The external cohort had limited events (19 EDR…)” | MATCH | 定性宣稱 |
| 35 | S9 限制 (card + data-cap) | 「multicenter prospective validation next」 | Discussion: “Prospective multicenter studies with larger event numbers will be essential…” | MATCH | 定性宣稱 |
| 36 | S10 結論 (h2 + data-cap) | 「Four routine variables. Consistent ~90% NPV.」 | Discussion: “The model maintained a negative predictive value of approximately 90% in both cohorts…”; Abstract: “…maintaining rule-out performance of approximately 90% across both cohorts…” | MATCH |  |
| 37 | S10 結論 (旁白 s10) | 「Consistent ninety percent negative predictive value」 | 同上（approximately 90%） | MATCH |  |
| 38 | S10 結論 (logo-line) | 「github.com/odafeng/Stage_III_Colon_EDR」 | Methods: “Source code is available at https://github.com/odafeng/Stage_III_Colon_EDR…” | MATCH |  |
| 39 | S10 結論 (署名) | 作者「Shih-Feng Huang, Chao-Wen Hsu, Yu-Hsun Chen, Chih-Chien Wu, Yi-Kai Kao」 | Manuscript 作者列相同（順序一致） | MATCH | 非數字但一併核對 |

## Manuscript 內部觀察（不影響影片判定，但供編輯參考）

1. **外部 cohort 總體 DFS 的 p 值兩處不同**：Results 內文為 Cox HR 之 p = **0.030**（“HR 1.83, 95% CI 1.06–3.16; p = 0.030”），而 Figure S4(A) 圖註為 log-rank p = **0.028**（“log-rank p = 0.028”）。兩者為不同檢定（Cox Wald vs log-rank），數值可並存但讀者易混淆。影片使用 **p = 0.030**，與 Results 內文完全一致。
2. **影片自身措辭不一致（非 Manuscript 問題）**：S2 畫面上寫「<18 mo」、caption 寫「≤18 mo」，Manuscript 用「within 18 months」（≈≤18 個月）。定義含義一致，無數值衝突。
3. 旁白中 0.68 / 77% / 90% / 0.63 為 0.680 / 77.4% / 89.8% / 0.633 的合法四捨五入，Manuscript 自身亦使用「approximately 90%」概括，判定 MATCH。

## 附註：任務提示中提及、但**並未出現於影片**的數字（無需比對）

以下數字存在於 Manuscript，但經全文搜尋（va_deck.html 與 scenes.json 皆無）**未出現在影片文字/旁白**，因此不是影片宣稱，僅列作交叉確認：

| 數字 | Manuscript 出處 | Manuscript 內一致性 |
|------|------------------|---------------------|
| threshold 0.12 | Methods「probability threshold of 0.12」；Figure S1 敘述 | 一致（cutoff 由 derivation 沿用至 external） |
| Brier 0.147（內部） | Results「Brier score 0.147」；Figure 2 圖註 | 一致 |
| Brier 0.130（外部） | Results「a Brier score of 0.130」 | 一致 |
| AUC 0.669（adapted MSKCC，內部） | Results「0.669 for the adapted MSKCC-variable model」 | 一致 |
| AUC 0.685（adapted MSKCC，外部） | Results「AUC 0.633 vs 0.685; p = 0.174」 | 一致 |
| DeLong p = 0.062（內部 vs AJCC） | Results「DeLong p = 0.062」 | 一致 |
| DeLong p = 0.744（內部 vs MSKCC） | Results「p = 0.744」 | 一致 |
| DeLong p = 0.174（外部 vs MSKCC） | Results「p = 0.174」 | 一致 |
| 1000 bootstrap | Methods「bootstrap resampling (1,000 replicates)」 | 一致 |
| 5-fold（外層）/ 3-fold（內層） | Methods「5-fold outer loop with an inner 3-fold randomized search」 | 一致 |
| MSKCC（比較模型） | Methods（Comparison with existing prediction models） | 一致 |
| 18.7% / 13.4%（event rate） | Discussion「18.7% and 13.4%, respectively」（62/331=18.7%、19/142=13.4%） | 一致 |
| stage IIIB 亞組：內部 log-rank p = 0.049、外部 p = 0.565 | Results「log-rank p = 0.049」；「p = 0.565」 | 一致（影片未提此二 p 值） |
| 外部 stage IIIB DFS events = 31 | Discussion「31 DFS events in stage IIIB」 | 一致（影片未提） |
| 內部 AJCC 比較值：sensitivity 45.2%、NPV 87.1%、specificity 45.7% | Results「77.4% vs. 45.2%」「89.8% vs. 87.1%」；S5「45.7%」 | 一致（影片未提） |

## 結論

影片（va_deck.html 畫面 + captions + scenes.json 旁白）中出現的所有數字/數據宣稱，與 Manuscript_final.docx 逐條核對後 **39/39 全數一致（MATCH）**；未發現 MISMATCH 或 UNVERIFIABLE 項目。影片中 AUC（0.680/0.633）、sensitivity（77.4%/68.4%）、NPV（89.8%/90.8%）、HR 1.83（p=0.030）、derivation 331 例/62 events、external 142 例/19 events、threshold 相關數字皆與論文相符。唯一需留意的是 Manuscript 內部 Figure S4(A) log-rank p=0.028 與 Results 內文 p=0.030 的檢定來源差異（影片採用內文值 0.030），以及影片自身「<18 mo vs ≤18 mo」的措辭小不一致——均不構成影片與論文之間的數字衝突。
