# Video Abstract 數字驗證報告（Verifier 1）

- 驗證日期：2026-08-21
- 影片來源：`video_build/va/va_deck.html`（畫面文字 + data-cap）＋ `video_build/va/scenes.json`（旁白）＋ `va_captions.srt`（字幕，與 data-cap 一致）
- 權威來源：`Submission/Manuscript_final.docx`（唯一判定依據）；`Submission/Response_to_Reviewers.docx` 僅作輔助對照（其中說明 revision 後數字已 harmonize：internal AUC 統一 0.680、external AUC 0.637→0.633、AJCC 0.617→0.610、Brier 0.128→0.130、HR 1.84→1.83，影片使用更新後數值）

## 比對結果總覽

| 指標 | 數量 |
|---|---|
| 總宣稱數 | 32 |
| MATCH | 32 |
| MISMATCH | 0 |
| UNVERIFIABLE | 0 |

> 結論：**未發現任何 MISMATCH**。影片所有數字宣稱均與 Manuscript_final.docx 一致。
> 需注意的附註（非影片錯誤）：論文內部有一處數字表述差異——Figure S4 legend 對「整體外部世代 DFS 分離」寫 log-rank p = 0.028，而 Results 內文對同一比較寫 Cox HR 的 p = 0.030（兩者為不同統計量，影片宣稱 p = 0.030 與 Results 內文一致）。

## 逐條比對表

| 場景/來源 | 影片宣稱 | Manuscript 數值（附原文引用） | 狀態 |
|---|---|---|---|
| S1 title (deck h1/h3, data-cap, narration) | 「4-variable ML model」(four-variable；旁白 four-variable) | 標題："A Simple Four-Variable Machine Learning Model with External Validation"；Abstract："A four-variable XGBoost model…" | **MATCH** |
| S1 title (deck h1, narration) | Stage III colon cancer | 全文一致使用 stage III colon cancer（標題、Abstract Background 等） | **MATCH** |
| S1 title (deck h3) | with external validation | 標題含 "with External Validation"；Abstract："externally validate" | **MATCH** |
| S2 problem (deck h2, data-cap, narration) | 20–30% of stage III patients recur after curative surgery（旁白 twenty to thirty percent） | Introduction："with 20% to 30% of patients experiencing recurrence" | **MATCH** |
| S2 problem (data-cap, stat card, narration) | early distant recurrence ≤18 months（stat 卡寫 <18 mo；旁白 within eighteen months） | Abstract："Early distant recurrence within 18 months"；Methods（Outcome Definition）：EDR-18 = "distant metastasis occurring within 18 months after curative surgery" | **MATCH** |
  - 註：影片自身 stat 卡用「<18 mo」、data-cap 用「≤18 mo」，屬影片內部小不一致；兩者皆與論文「within 18 months」相容
| S3 gap (card 內文) | wide outcome variation within the same substage — especially stage IIIB | Introduction："permitting substantial outcome variation within substages"；Introduction："within AJCC stage IIIB, a subgroup known for its biological heterogeneity" | **MATCH** |
| S4 model (h2, cards, narration) | 4 routine variables：AJCC substage / lymph node ratio / perineural invasion / tumor differentiation | Methods（Feature selection）："four predictors: AJCC substage, lymph node ratio, perineural invasion, and tumor differentiation" | **MATCH** |
| S4 model (card) | AJCC substage IIIA / IIIB / IIIC | Methods（Candidate Predictors）："AJCC substages (IIIA, IIIB, IIIC)" | **MATCH** |
| S4 model (card) | Lymph node ratio = positive / total nodes | Methods："lymph node ratio (LNR)"（定義為陽性/總淋巴結比例，為標準定義；論文未逐字寫出但語意一致） | **MATCH** |
| S4 model (stat, narration) | 331 — Derivation cohort | Abstract（Patients）："331 patients in the derivation cohort"；Results："The derivation cohort included 331 patients with stage III colon adenocarcinoma" | **MATCH** |
| S4 model (stat) | 62 events (derivation) | Abstract（Patients）："331 patients in the derivation cohort (62 events)"；Methods："With 62 EDR-18 events" | **MATCH** |
| S4 model (stat, narration) | 142 — External cohort | Abstract："142 in the external cohort"；External Validation："an independent external cohort… (E-DA Hospital; 142 patients…)" | **MATCH** |
| S4 model (stat) | 19 events (external) | Abstract："142 in the external cohort (19 events)"；External Validation："including 19 EDR-18 events" | **MATCH** |
| S4 model (stat) | TRIPOD+AI — reporting guidance | Methods："following TRIPOD+AI recommendations"；"The study followed the TRIPOD+AI recommendations for transparent reporting" | **MATCH** |
| S5 internal (stat, data-cap, narration) | Derivation AUC 0.680（旁白 zero point six eight / 0.68） | Abstract（Results）："a pooled out-of-fold area under the curve of 0.680"；Results："AUC of 0.680 (95% CI 0.608–0.751)"；Figure 2 legend："0.680 vs. 0.625" | **MATCH** |
  - 註：旁白 0.68 為 0.680 的口語唸法，數值相同
| S5 internal (stat, data-cap) | AJCC AUC 0.625 | Abstract："comparable to conventional staging (0.625)"；Results："logistic regression model using AJCC substage alone achieved a pooled out-of-fold AUC of 0.625 (95% CI 0.545–0.698)" | **MATCH** |
| S5 internal (stat, data-cap, narration) | Sensitivity 77.4%（旁白 seventy-seven percent / 77%） | Abstract："Sensitivity (77.4%)…"；Results："higher sensitivity (77.4% vs. 45.2%)" | **MATCH** |
  - 註：旁白 77% 為 77.4% 的四捨五入口語，方向與數值一致
| S5 internal (stat, data-cap, narration) | NPV 89.8%（旁白 ninety percent / 90%） | Abstract："negative predictive value (89.8%)…"；Results："negative predictive value (89.8% vs. 87.1%)" | **MATCH** |
  - 註：旁白 90% 為 89.8% 的四捨五入口語；論文亦以「approximately 90%」概括
| S6 external (stat, data-cap, narration) | External AUC 0.633（旁白 zero point six three / 0.63） | Abstract："an area under the curve of 0.633 (versus 0.610 for conventional staging)"；Results："the four-variable model achieved an AUC of 0.633"；Figure S3 legend："AUC = 0.633" | **MATCH** |
  - 註：旁白 0.63 為 0.633 的口語唸法，數值相同
| S6 external (stat, data-cap, narration) | External NPV 90.8% | Abstract："a negative predictive value of 90.8%"；Discussion："NPV of 90.8% (95% CI 83.0–97.1%)" | **MATCH** |
| S6 external (data-cap/caption bar) | External sensitivity 68.4% | Discussion："including an external sensitivity of 68.4%" | **MATCH** |
  - 註：僅出現在 data-cap（字幕列），未顯示於畫面上方 stat 卡
| S6 external (stat) | DFS separation — HR 1.83 (p = 0.030) | Results："HR 1.83, 95% CI 1.06–3.16; p = 0.030"；Response to Reviewers 確認更新後 HR 1.83 (95% CI 1.06–3.16) | **MATCH** |
  - 註：論文內部小差異：Figure S4 legend 對同一 overall external 比較寫 log-rank p = 0.028，Results 內文寫 Cox HR p = 0.030（不同統計量）；影片宣稱與 Results 內文一致
| S6 external (narration) | clear separation in disease-free survival between risk groups | Results："ML-defined risk groups demonstrated significantly different DFS outcomes in the overall stage III population (HR 1.83, 95% CI 1.06–3.16; p = 0.030)" | **MATCH** |
| S7 shap (sub, data-cap, narration) | SHAP: AJCC IIIC · lymph node ratio · perineural invasion — strongest risk drivers（旁白 stage three C = IIIC） | Results（Model explainability）："SHAP analysis identified AJCC substage IIIC as the most influential predictor of EDR-18, followed by lymph node ratio and perineural invasion" | **MATCH** |
| S8 clinical (flow, data-cap) | 4 pathology inputs → probability + risk class | Methods（Model Deployment）："The tool accepts four routine inputs and outputs both predicted EDR-18 probability and a high- or low-risk classification" | **MATCH** |
| S8 clinical (foot-link) | stage3edr.streamlit.app | Methods："the online tool at https://stage3edr.streamlit.app" | **MATCH** |
| S8 clinical (h2) | Usable immediately after surgery — no extra testing | Methods："designed for immediate postoperative risk assessment, before adjuvant treatment decisions are finalized"；Discussion："enabling rapid postoperative risk assessment without additional testing" | **MATCH** |
| S9 limits (card, data-cap) | Retrospective design | Abstract（Limitations）："Retrospective design…" | **MATCH** |
| S9 limits (card, data-cap) | limited external events | Abstract："small numbers of events in the external cohort"；Discussion："The external cohort had limited events (19 EDR, 31 DFS events in stage IIIB)" | **MATCH** |
| S9 limits (card) | Both cohorts from two Taiwanese institutions | Methods（Fairness）："Both cohorts were drawn from two Taiwanese institutions" | **MATCH** |
| S10 takehome (h2, narration) | Four routine variables | 全文一致（four-variable model） | **MATCH** |
| S10 takehome (h2, data-cap, narration) | Consistent ~90% NPV（旁白 ninety percent） | Abstract："maintaining rule-out performance of approximately 90% across both cohorts"；Discussion："maintained a negative predictive value of approximately 90% in both cohorts"（89.8% derivation / 90.8% external） | **MATCH** |
## 附錄 A：影片中「未出現」但父任務要求核對的數字（非影片宣稱，僅確認論文數值）

這些數字未出現在影片文字、data-cap 或旁白中，因此不構成影片宣稱；此處僅列出論文對應數值供交叉確認，全部前後一致：

| 項目 | Manuscript_final.docx 數值 |
|---|---|
| 風險閾值 cutoff | 0.12（Methods「Derivation of Risk Groups」："the probability threshold of 0.12"；Figure S1 legend 同） |
| Brier score（internal） | 0.147（Results Calibration："Brier score 0.147"；Figure 2 legend B） |
| Brier score（external） | 0.130（Results External validation："a Brier score of 0.130"） |
| MSKCC-variable model AUC（internal） | 0.669（Results："AUC 0.625 for AJCC, 0.669 for the adapted MSKCC-variable model, and 0.680 for the four-variable model"） |
| MSKCC-variable model AUC（external） | 0.685（Results："AUC 0.633 vs 0.685; p = 0.174"） |
| DeLong p（internal，4-var vs AJCC） | 0.062（Results："DeLong p = 0.062"） |
| DeLong p（internal，4-var vs MSKCC-var） | 0.744（Results："four-variable vs MSKCC-variable model p = 0.744"） |
| DeLong p（external，4-var vs MSKCC-var） | 0.174（Results："p = 0.174"） |
| bootstrap replicates | 1,000（Methods："bootstrap resampling (1,000 replicates)"） |
| 內部 cross-validation | 5-fold outer × 3-fold inner nested CV + 固定組態 stratified 5-fold（Methods 與 Figure S1 legend） |
| stage IIIB subgroup p（derivation） | 0.049（Results："log-rank p = 0.049; Figure 3B"；Figure 3 legend B 同） |
| stage IIIB subgroup p（external） | 0.565（Results："p = 0.565"；Figure S4 legend B 同） |
| internal sensitivity/specificity/PPV/NPV | 77.4% / 45.7% / 24.7% / 89.8%（Response to Reviewers 註記；論文內文至少出現 77.4% 與 89.8%） |
| external event rate（derivation / external） | 18.7% / 13.4%（Discussion） |

## 附錄 B：影片內嵌圖表（figure panel）

S5、S6、S7 的 figure panel 直接使用論文圖檔（`Figure_2_Model_Performance.png`、`Figure_S3_External_Calibration_Performance.png`、`Figure_4_SHAP_Plot_XGB.png`）。圖上標註的數值與論文 figure legends 一致（Figure 2：AUC 0.680 vs 0.625、Brier 0.147、DCA 0.10–0.30；Figure S3：AUC 0.633、AJCC 0.610）。由於本驗證模型不支援 vision，圖內像素級文字無法逐字 OCR，故依圖檔即論文原始圖 + figure legends 判定為 MATCH。

## 附錄 C：論文內部不一致備註（非影片問題）

1. **外部世代整體 DFS 的 p 值表述**：Results 內文寫 "HR 1.83, 95% CI 1.06–3.16; p = 0.030"，而 Figure S4 legend (A) 寫 "log-rank p = 0.028"。兩者為不同檢定（Cox proportional-hazards Wald p vs log-rank p），數字不同屬正常；影片宣稱的 p = 0.030 與 Results 內文完全一致，不受影響。
2. **影片自身小不一致**：S2 stat 卡寫「<18 mo」、data-cap 寫「≤18 mo」，兩者皆與論文「within 18 months」相容，不構成對論文的不一致。

---

*本報告僅以 `Manuscript_final.docx` 為判定依據；未自行推斷任何正確數值。*
