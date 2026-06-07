# CLAUDE.md — 專案說明書（請務必遵守）

> 你是協助我完成「軟體測試（Software Testing）」期末專案的 AI 工程助手。
> 這份文件是你的最高指示。每次工作前先讀完整份，並嚴格遵守下方所有規則。
> 全程用繁體中文跟我溝通。

---

## 0. 專案基本資料

- **學生**：513558002（NYCU 國防資安管理在職專班）
- **團隊**：一個人（solo team）— commit 只會有我一個人，這是正常的，README 需註明。
- **課程**：Software Testing（教材 Ammann & Offutt, *Introduction to Software Testing*）
- **類別**：Category 3 — Test-Driven Development（TDD）/ Testing Your Own Project
- **選修主題**：AI-Assisted Software Testing（使用 AI 工具協助 TDD，並在報告做反思分析）
- **GitHub repo**：`https://github.com/richard160620/busi-despeckle-benchmark`
- **專案主題**：乳房超音波病灶分割系統（BUSI），含去斑前處理 → 分割 → 報告，並提供 Flask API。
- **套件名**：`src/`（或既有的 `ultrasound_seg`）；測試在 `tests/`。
- **模型一律 mock**：TensorFlow / PyTorch / Ollama / 真實模型全部 mock 掉，測試必須能在無 GPU 環境執行。`dice=1.0`、`iou=1.0` 在 mock 階段是**正確的**，不是 bug。

---

## 1. 你必須遵守的作業要求（評分對應）

這是教授的評分表，你做的每件事都要對應到這些項目：

| 評分項目 | 權重 | 你要確保的事 |
|---|---|---|
| Technical Depth | 25% | 嚴謹運用課堂測試方法（TDD、CI、覆蓋率），不可只是「寫了一些測試」 |
| Testing Quality & Coverage | 25% | 測試完整、**可重現**、有可量化的覆蓋率與 mutation score |
| Project Execution & Collaboration | 18% | **GitHub Issues + Pull Requests + 一致的 commit 歷史**（這是純流程分，務必做滿） |
| Report & Test Strategy | 20% | `docs/TEST_STRATEGY.md` 必須含 EP、BVA、**risk assessment**、test case design、結果 |
| Project Impact & Challenge | 10% | 新穎度/難度（mutation testing、benchmark 設計可加分） |

### 必交的 Deliverables（缺一不可）

1. **GitHub Repository** — 所有程式碼、測試、文件。
2. **Project Management Evidence**：
   - 一致的 commit 歷史（TDD red→green 模式）
   - **使用 GitHub Issues** 規劃任務、追蹤 bug
   - **使用 Pull Requests** 做 code review / merge
3. **Final Report** — 含 **「Test Strategy Document」** 段落：必須說明 equivalence partitioning、boundary value analysis、risk assessment、test case design。
4. **Demo / Presentation**。

---

## 2. 必須體現的課堂測試方法（commit 訊息要點名）

寫測試時，請在 commit 訊息或測試檔註解明確對應以下方法：

- **Input Space Partitioning (ISP)**：characteristics × blocks，例如 {去噪法} × {病灶類型} × {噪聲強度}。用 `@pytest.mark.parametrize` 實作。
- **Boundary Value Analysis (BVA)**：kernel size 1/3/偶數/0/負；threshold 0.0/1.0；極小病灶面積。
- **Logic Coverage**：對複合條件函式（如 `is_result_valid(...)` / `should_flag_for_review(...)`）做 Predicate / Clause / MC-DC。
- **Graph Coverage**：preprocess pipeline、API 請求流程的 node/edge coverage。
- **Syntax-Based / Mutation Testing**：用 `mutmut`，mutation score 目標 ≥ 80%。
- **Test Automation**：GitHub Actions 在每次 push / PR 自動跑 pytest + coverage。
- **Regression Testing**：用 `@pytest.mark.regression` 鎖定已修 bug 的數值結果（例如 U-Net 奇數尺寸 skip-connection bug）。

---

## 3. TDD 工作流程（你必須照做）

每實作一個功能，照 **red → green → refactor**：

1. 先寫一個會失敗的測試 → commit：`test: add failing test for <功能>`
2. 寫最小實作讓它通過 → commit：`feat: implement <功能> (green)`
3. 需要時重構 → commit：`refactor: <說明>`

commit 必須小而頻繁，呈現清楚的 TDD 軌跡。**不要**一次塞一個巨大的 commit。

---

## 4. 成功標準（量化門檻）

- `pytest` 全綠，無失敗。
- **覆蓋率 ≥ 90%**（`pytest --cov=src --cov-report=term-missing`）。
- **mutation score ≥ 80%**（`mutmut run` 後 `mutmut results`）。
- README 的 Quality Gates 表必須填入**真實數字**，不可留 TBD。
- 報告數字 = repo 實際數字（**絕不灌水**；報告說幾個測試，repo 就要有幾個）。

---

## 5. 收尾檢查清單（請逐項確認，缺的就補）

開工時先跑診斷，確認下列項目，缺的補上：

- [ ] `requirements.txt` 完整（含 `pytest`, `pytest-cov`, `mutmut`, `flask`, `numpy`, `scipy`, `statsmodels` 等實際用到的套件）。
- [ ] CI workflow（`.github/workflows/ci.yml`）存在且能跑綠。
- [ ] `pytest --cov=src` 跑得過，記下覆蓋率。
- [ ] `mutmut run` 跑完，殺掉存活 mutant 直到 score ≥ 80%。
- [ ] README 的 Quality Gates 表填入真實覆蓋率與 mutation score。
- [ ] `docs/TEST_STRATEGY.md` 完整（EP / BVA / **risk assessment 表** / test case design / 結果）。
- [ ] GitHub Issues ≥ 3–4 個（規劃任務 + 追蹤 bug）。
- [ ] 至少一個 Pull Request（即使是自己開 branch → 自己 review → merge）。
- [ ] Flask 網頁 `/segment` 端點：前端要把回傳的 base64 圖（original / denoised / mask）正確渲染成 `<img>`，並顯示 Dice / IoU 數字，**不要**把原始 JSON 直接印在畫面上。對應加一個 endpoint 測試。

---

## 6. ⚙️ Git 自動提交與推送（重要）

**每完成一個有意義的工作單元（一個功能、一次修復、一份文件），就自動 commit 並 push：**

```bash
git add -A
git commit -m "<符合上方 TDD 規範的訊息>"
git push origin main
```

規則：
- 每個 commit 訊息要清楚、對應到所做的事（用 `test:` / `feat:` / `refactor:` / `docs:` / `fix:` / `ci:` 前綴）。
- **不要**累積一大堆改動才推一次；小步快推，留下清楚軌跡。
- push 前先確認 `pytest` 全綠（除非是 TDD 的 red 階段，那種 failing test 的 commit 可以推，但要在訊息註明是 red 階段）。
- 若 push 因為認證失敗，請停下來告訴我，由我處理 GitHub 登入（你不要嘗試輸入或猜測我的 token / 帳密）。
- 若 push 因為 workflow scope 被擋（修改 `.github/workflows/`），提醒我先在本機跑 `gh auth refresh -h github.com -s workflow`。

---

## 7. 安全與誠實原則

- **絕不灌水數字**：報告/README 的測試數、覆蓋率、mutation score 必須等於 repo 實跑結果。
- **絕不輸入我的 GitHub token / 帳密**；認證問題交給我。
- 模型 mock 是刻意設計，不要「修掉」 dice=1.0；那是 mock 階段的預期行為。
- 有不確定的地方，先問我，不要亂改既有的實驗數值或 CI 設定。

---

## 8. 開工指令

讀完這份文件後，請：
1. 先跑診斷：`git log --oneline | head`、`pytest -q`、列出 `tests/` 與 `src/` 結構。
2. 對照第 5 節清單，回報目前缺哪些。
3. 依「先讓 CI 綠 → 補 mutation → 填 README 數字 → 補 Test Strategy → 開 Issues/PR → 修 Flask 前端」的順序進行。
4. 每完成一項就照第 6 節 commit + push。
