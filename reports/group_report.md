# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Mọi con số dưới đây lấy từ file do script sinh ra. Số trên **eval** chỉ có khi giảng viên chạy
> `make eval-baseline eval-poc compare` với overlay `gt/`. Repo học viên không có nhãn eval, nên bảng chính thức
> (mục 04) để trống cho tới lúc đó. Mọi số trong báo cáo là số **nội bộ trên dev** (nhãn tay), đã ghi rõ là lạc quan.

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | Không có (làm cá nhân) |
| Challenge | T02 · Raw QC Triage: ảnh mờ, lóa, thiếu sáng, nén |
| Thành viên + vai trò | Pham Huynh (2A202603135): làm cá nhân, đảm nhận mọi vai trò (integrator, algorithm, QA/metric, report) |
| Git URL | https://github.com/phamhuynhmk1-png/K5-Day6-L2-PhamHuynh-2A202603135-DataGovernanceAndPrivacy |
| Tag `demo-180m` (commit) | `demo-180m` → 72c9f36 |
| Tag `final-24h` (commit) | `final-24h` → commit cuối trên main (xem `git show final-24h`) |

## 01 · Problem statement

- **Ai bị ảnh hưởng:** đội huấn luyện model perception. Ảnh lỗi (mờ, tối, lóa, vỡ nén) lọt vào tập train/eval sẽ làm
  nhiễu nhãn và đánh giá sai model. Ngược lại, loại nhầm ảnh tốt (nhất là ảnh **đêm** và ảnh **cam_B**) làm mất đúng
  những điều kiện khó mà model cần học.
- **Điểm nghẽn:** reviewer chỉ có thời gian xem **B = 20** ảnh mỗi lô. Cần xếp hạng để 20 ảnh đó chứa nhiều lỗi thật nhất.
- **Đầu vào → đầu ra:** `images.csv` + 96 ảnh JPEG → `ranked_review.csv` (`image_id, score, pred_defect_type, pred_action`),
  `review_queue.csv`, `contact_sheet.html`, `audit_log.jsonl`, `decision.json`.
- **Tiêu chí quyết định cho lô eval (chốt trước khi chạy):**
  - **HOLD** nếu còn ảnh trong queue chưa review, hoặc có ảnh bị `escalate`.
  - **REWORK** nếu tỉ lệ ảnh cần chụp lại vượt `max_recapture_rate` = 25%.
  - **GO** (kèm danh sách recapture) nếu không rơi vào hai trường hợp trên.
- **Giả định cần domain owner xác nhận:** ngưỡng 25% recapture là giả định của nhóm, chưa có SLA thật. "Lỗi" được hiểu
  là ảnh không dùng được cho training, không xét lỗi nhẹ chấp nhận được.

## 02 · Data & provenance

| Split | N | Nguồn | License | Synthetic? |
|---|---|---|---|---|
| dev | 56 ảnh (14 cảnh × 4) | `fixtures/t02_raw_qc` (generator seed 42 của giảng viên) | CC0-1.0 | Có, không có PII thật |
| eval | 40 ảnh (10 cảnh × 4); 16 ảnh night | như trên | CC0-1.0 | Có |

- **Schema:** `image_id, path, camera_id ∈ {cam_A, cam_B}, light ∈ {day, dusk, night}, split`. Lớp lỗi:
  `none, blur, dark, glare, compression`.
- **Split** chia **theo cảnh gốc** (do đề cấp sẵn), nên các biến thể của một cảnh không bị rò rỉ giữa dev và eval.
  Seed 42. Pipeline không có bước ngẫu nhiên, chạy 2 lần ra file giống hệt (có test).
- **`manifest.csv` verified (sha256): có.** `load()` kiểm tra hash và dừng với exit 2 nếu lệch (đã thử bằng cách sửa `images.csv`).
  - **Sự cố môi trường:** Git trên Windows (`core.autocrlf=true`) đổi LF thành CRLF trong fixtures, làm lệch hash của
    cả 14 track. Đã sửa bằng `.gitattributes` (`fixtures/** -text`) rồi checkout lại.
- **Gold set nội bộ:** `configs/t02_dev_manual_labels.csv`, gồm 56 ảnh dev do Pham Huynh gán nhãn bằng mắt (28 none,
  10 compression, 8 blur, 6 dark, 4 glare).
  - Một nhãn đã sửa sau khi phóng to: img_08_2 nhìn ở cỡ thumbnail tưởng là blur, nhưng zoom 3× thấy ringing JPEG,
    nên đổi thành compression.
  - Không gán nhãn ảnh eval.
- **Không dùng dữ liệu thật.**

## 03 · Research & design

- **Cơ sở chọn kỹ thuật:**
  - Gợi ý của handout: kết hợp sharpness, histogram, saturated fraction và JPEG blockiness; chuẩn hoá theo `light`/`camera_id`.
  - ACDC (ETH Zürich, dataset điều kiện bất lợi): cho thấy ảnh đêm/sương là **điều kiện hợp lệ**, không phải lỗi.
    Vì vậy độ tối phải so với chuẩn của **cùng điều kiện ánh sáng**.
  - Đo blockiness theo lưới 8×8 là cách kinh điển phát hiện artefact nén JPEG (DCT block).
- **Thiết kế: mỗi họ lỗi một đặc trưng** (`src/t02_qc.py`). Mỗi đặc trưng chọn để không vướng các bẫy trong đề:

| Họ lỗi | Đặc trưng | Bẫy được né |
|---|---|---|
| blur | Laplacian variance toàn ảnh, ngưỡng thấp (31.5) | cam_B ống kính mềm (ảnh sạch cam_B có Laplacian ≥ 166) |
| dark | độ sáng trung bình / độ sáng chuẩn của cùng `light` (day 120, dusk 89, night 32) | ảnh đêm hợp lệ không bị coi là "tối" |
| glare | diện tích vùng cháy (> 235) liền khối lớn nhất | lóa cục bộ ở biển số trong ảnh vẫn nét |
| compression | tỉ lệ gradient tại biên khối 8×8 / bên trong khối | độc lập với độ sáng và độ nét |

- **Cách gộp điểm:**
  - Mỗi đặc trưng đổi thành **margin có dấu theo thang log**: bằng 0 tại ngưỡng, +1 tại ảnh lỗi dev gần ranh giới
    nhất, −1 tại ảnh sạch dev gần ranh giới nhất.
  - `score` = max margin. `pred_defect_type` = họ đạt max.
  - Cách này không cần học trọng số trên 56 ảnh. Một lỗi mạnh là đủ để ảnh được ưu tiên review.
- **Data flow:**
  ```
  images.csv + manifest (sha256) → 01 load → 02 process (4 đặc trưng → margin → score)
    → 03 rank (top-20 của eval → review_queue.csv + contact_sheet.html)
    → 04 review (review_queue_reviewed.csv → override + audit_log.jsonl)
    → 05 decide (GO / HOLD / REWORK + reason_code + owner) → ranked_review.csv, decision.json
  ```
- **Thành phần offline:** chỉ numpy, scipy, Pillow, pandas. Không có model hay weights, không gọi mạng.
- **Trade-off:**
  - Chọn đặc trưng thủ công dễ giải thích, thay vì classifier học từ 56 ảnh (dễ overfit).
  - Đổi lại, phải tự lường trước từng kiểu lỗi, và một kiểu lỗi lạ sẽ bị bỏ sót (xem failure case 1).

## 04 · Baseline vs PoC

- **Baseline** (`src/baseline_t02.py`):
  - Theo đúng handout: **một ngưỡng Laplacian toàn cục** cho mọi ảnh.
  - `score = log(759 / lap_var)`. Ngưỡng 758.8 tối đa F1 trên dev (`tools/calibrate_t02.py`).
  - Dùng chung code load, queue, review, decide với PoC, nên chỉ khác ở phần chấm điểm.
- **PoC** (`src/pipeline.py` + `src/t02_qc.py`): 4 họ đặc trưng như mục 03. Ngưỡng chốt trong `configs/lab.yaml`.
- **Cùng điều kiện:** cùng pool eval 40 ảnh, cùng B = 20, cùng seed 42, cùng evaluator.

**Bảng 1 · Chính thức trên eval:** _chờ giảng viên chạy_ `make eval-baseline eval-poc compare TRACK=t02`
(cần overlay `gt/`). Repo học viên chạy lệnh này sẽ exit 2. Không gõ tay số vào đây.

**Bảng 1b · Nội bộ trên DEV** (nhãn tay, B = 20, `python -m tools.dev_eval_t02` → `reports/t02_dev_internal_eval.txt`).
⚠️ Ngưỡng được chọn trên chính tập này nên số **lạc quan**, chỉ dùng để so sánh tương đối:

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Primary: Defect Recall@20 (image) | 0.571 (16/28), Wilson 95% [0.391, 0.735] | **0.714 (20/28)**, [0.529, 0.847] | +0.143. Trần = 20/28 = 0.714 vì có 28 lỗi mà chỉ có 20 lượt xem |
| Precision@20 | 0.800 (16/20), [0.584, 0.919] | **1.000 (20/20)**, [0.839, 1.000] | +0.200 |
| Hard slice: ảnh sạch cam_B lọt top-20 | 4 (img_12_0, 09_0, 21_0, 00_0) | **0** | bẫy "cam_B quang học mềm" |
| Recall theo loại: glare / compression | 0.00 / 0.30 | 0.50 / 0.90 | baseline mù với lỗi không làm giảm độ nét |
| Defect-type accuracy (ảnh lỗi) | 0.286 (chỉ biết nói "blur") | **1.000** | +0.714 |
| Runtime toàn bộ run (96 ảnh) | 0.43 s | 1.16 s (3.45 ms/ảnh ở bước process) | CPU AMD64 Family 25, Windows 11, Python 3.12 |

Các khoảng Wilson của Recall chồng lên nhau (N = 28). Chênh lệch Recall trên dev **chưa có ý nghĩa thống kê**.
Lợi thế rõ hơn nằm ở precision, ở khả năng đoán đúng loại lỗi và ở số false alarm cam_B.

**Ablation trên DEV** (`python -m tools.ablation_t02` → `reports/t02_ablation_dev.md`). Mỗi dòng tắt một họ đặc trưng:

| Biến thể | Recall@20 | Precision@20 | Type acc | False alarm top-20 |
|---|---|---|---|---|
| PoC đủ 4 họ | 0.714 | 1.000 | 1.000 | 0 |
| bỏ blur | 0.714 | 1.000 | 0.714 | 0 |
| bỏ dark | 0.714 | 1.000 | 0.786 | 0 |
| bỏ glare | 0.714 | 1.000 | 0.857 | 0 |
| bỏ compression | 0.679 | 0.950 | 0.643 | 1 |

**Đọc trung thực:**
- Với 28 lỗi và chỉ 20 lượt xem, bỏ một họ (trừ compression) vẫn còn đủ ảnh lỗi để lấp top-20, nên Recall@20 trên dev
  **không đo được** đóng góp của blur, dark và glare.
- Đóng góp của các họ này thể hiện ở **type accuracy** và ở cột `pred_action` (danh sách recapture đầy đủ).
- Compression là họ duy nhất mà thiếu đi thì Recall giảm.

## 05 · Evaluation protocol

- **Metric:** xem `reports/metric_contract.md`. Recall@B = #ảnh `is_defect = 1` trong top-B / #ảnh lỗi của eval.
  Đơn vị là ảnh, ghép theo `image_id`. Hoà điểm giữ thứ tự file.
- **Ngưỡng:**
  - Chọn **chỉ trên dev** (nhãn tay) bằng `tools/calibrate_t02.py`.
  - Với mỗi họ lỗi, ngưỡng là điểm giữa theo thang log giữa ảnh sạch gần nhất và ảnh lỗi gần nhất. Cả 4 họ đều tách
    được hoàn toàn trên dev (`reports/t02_calibration.json`).
  - Chốt vào `configs/lab.yaml` ngày 2026-11-14, **trước** khi xem bất kỳ ảnh eval nào.
- **Độ bất định:** Wilson 95% CI (`src/metrics/estimation.wilson_ci`), báo kèm numerator/denominator.
- **Lệnh evaluate chính thức (giảng viên chạy):**
  `python -m src.evaluate --track t02 --pred out/poc/ranked_review.csv --run-name poc --out reports/metrics_poc.json`

## 06 · Result slices & failure cases

Slice trên DEV (top-20, nội bộ):

| Slice | N | Baseline recall / false alarm | PoC recall / false alarm | Ghi chú |
|---|---|---|---|---|
| light=day | 36 | 0.611 / 0.167 | 0.667 / 0.000 | false alarm của baseline đều là cam_B |
| light=dusk | 16 | 0.500 / 0.125 | 0.875 / 0.000 | false alarm của baseline là cam_B (img_12_0) |
| light=night | 4 | 0.500 / 0.000 | 0.500 / 0.000 | chỉ 4 ảnh đêm ở dev, trong khi eval có 16: rủi ro lớn nhất |

**Failure case 1: glare ban đêm bị bỏ sót** (img_03_2, img_06_3, eval, cam_B night).
- Ảnh có quầng sáng rất lớn trên nền đêm, nhưng đỉnh sáng **không vượt 235**, nên đặc trưng glare (ngưỡng tuyệt đối)
  thấy vùng cháy nhỏ. Cả hai ảnh có score âm và **không vào queue**.
- **Root cause:** dev chỉ có **1** ảnh glare ban đêm (img_18_2), và quầng sáng của ảnh đó cháy trắng hẳn (404 px > 235).
  Ngưỡng tuyệt đối vì vậy trông như đủ dùng.
- Phát hiện khi xem ảnh eval **sau khi đã chốt ngưỡng**. **Không sửa ngưỡng** vì như vậy là tune trên eval.
- **Đề xuất v2:** đo vùng sáng **tương đối** với độ sáng chuẩn của `light` (ví dụ pixel > ref_light + k·σ). Cần kiểm
  chứng trên một heldout mới, không phải eval này.

**Failure case 2: nén JPEG rất nhẹ bị bỏ sót** (img_19_2, eval, cam_A day).
- Blockiness của ảnh này nằm dưới ngưỡng 1.77.
- **Root cause:** ngưỡng là điểm giữa giữa ảnh nén nhẹ nhất ở dev (img_08_2, 1.886) và ảnh sạch có blockiness cao nhất
  (img_12_0, 1.664). Ảnh nén còn nhẹ hơn img_08_2 thì rơi vào vùng sạch.
- Ở mức này, reviewer người cũng phải zoom 3× mới thấy, nên có thể chấp nhận được tuỳ định nghĩa "không dùng được".

**Ca mơ hồ về loại lỗi:** img_15_3 vừa tối vừa hơi mờ (`blur = +0.46, dark = +3.54`). Máy chọn dark (họ trội), reviewer
đồng ý. Evaluator chỉ chấp nhận một nhãn, nên ảnh mang nhiều lỗi có thể bị tính sai type.

**Các failure mode handout yêu cầu, đã xử lý và có test (`tests/test_t02_pipeline.py`):**
- Ảnh đêm sạch không bị coi là dark (`test_clean_night_image_is_not_flagged_dark`).
- cam_B mềm không bị coi là blur (`test_soft_cam_b_optics_not_flagged_blur`).
- Lóa cục bộ trong ảnh nét được bắt (`test_local_glare_in_sharp_image_is_caught`).

## 07 · Human & decision

- **Review queue:** `out/poc/review_queue.csv`, gồm top-20 theo `score` trong pool **eval**. Cột `suggested_action`
  có dạng `RECAPTURE:<type>`, cột `reason` liệt kê margin dương.
  - Hạng 1–17 là ảnh máy gắn cờ. Hạng 18–20 là ảnh máy cho là sạch nhưng kém chắc chắn nhất, để reviewer có cơ hội bắt lỗi bị bỏ sót.
  - Có giao diện `contact_sheet.html` (offline, nút Export CSV).
- **Ý nghĩa quyết định của reviewer** (`src/pipeline.py::apply_review`):

  | Quyết định | Tác động |
  |---|---|
  | `approve` | giữ nhãn máy |
  | `reject` | false alarm, chuyển thành KEEP / none |
  | `edit` | `override_value` ∈ {blur, dark, glare, compression, none} |
  | `escalate` | chặn GO, decision thành HOLD |
  | `skip` | không làm gì |

  - Dòng có quyết định mà thiếu tên reviewer, hoặc `override_value` không hợp lệ, thì pipeline dừng với exit 2 (có test).
  - Mỗi thay đổi được ghi vào `audit_log.jsonl` (actor, before, after, reason).
  - Cột `score` gốc **không bao giờ bị sửa**, nên ranking vẫn kiểm toán được.
- **Lượt review đã thực hiện** (`reports/review_queue.csv`, bản copy của `out/poc/review_queue_reviewed.csv`):
  - Người duyệt: Pham Huynh. Xem từng ảnh, phóng to 3× các ca sát ngưỡng (img_03_3, img_16_3, img_05_3, img_06_0).
  - Kết quả: **20/20 approve**. 17 ảnh lỗi đều đúng loại, 3 ảnh sạch giữ nguyên.
  - Reviewer chỉ duyệt được ảnh **trong** queue. Ba ảnh bị bỏ sót (failure case 1–2) nằm ngoài queue, nên quy trình hiện
    tại không cứu được. Đó là lý do cần cải tiến v2 ở phần chấm điểm, không phải ở bước review.
- **Decision** (`out/poc/decision.json`, `reports/decision_log.csv`):

  | Thời điểm | Status | reason_code | Chi tiết |
  |---|---|---|---|
  | Trước review | HOLD | `PENDING_HUMAN_REVIEW` | 20/20 ảnh chưa duyệt |
  | Sau review | **REWORK** | `RECAPTURE_RATE_ABOVE_POLICY` | 17/40 ảnh eval (42.5%) cần chụp lại, vượt policy 25%: 7 compression, 4 dark, 4 blur, 2 glare |

  Owner: Pham Huynh (Data QC owner).
- **Khi nào không tự động release:** còn ảnh trong queue chưa duyệt, có `escalate`, hoặc tỉ lệ recapture vượt policy.
  Pipeline không bao giờ ra GO khi chưa có người duyệt.

## 08 · Reproduce & tests

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Linux/mac: .venv/bin/python
.venv/Scripts/python -m tools.calibrate_t02                     # in lại ngưỡng từ dev (đã chốt trong configs/lab.yaml)
.venv/Scripts/python -m src.main --config configs/lab.yaml --track t02 --pipeline src.baseline_t02 --output out/baseline --run-name baseline --offline --seed 42
.venv/Scripts/python -m src.main --config configs/lab.yaml --track t02 --output out/poc --run-name poc --offline --seed 42
.venv/Scripts/python -m tools.dev_eval_t02                      # số nội bộ trên dev
.venv/Scripts/python -m tools.ablation_t02                      # ablation trên dev
.venv/Scripts/python -m pytest -q
# giảng viên (cần overlay gt/): make eval-baseline eval-poc compare TRACK=t02 PY=.venv/Scripts/python
```

- **`pytest`:** `62 passed, 39 skipped`. 39 test bị skip là test cần `gt/` của giảng viên.
  15 test của nhóm nằm trong `tests/test_t02_pipeline.py`:
  - đủ 4 loại lỗi được phát hiện đúng (inject bằng `common.corruptions`)
  - 3 failure mode của handout
  - E2E chạy 2 lần cho file giống hệt (idempotent)
  - baseline chạy trên cùng pool
  - review thiếu tên thì fail, override sai giá trị thì fail
  - override được áp và ghi audit
  - thiếu ngưỡng thì fail rõ ràng
- `tests/test_cli.py::test_template_pipeline_fails_clearly_not_silently` đổi từ t02 sang t11: test gốc khẳng định template
  t02 **chưa** được cài, điều không còn đúng sau khi cài t02. Ý nghĩa test giữ nguyên: track chưa cài phải trả exit 3.
- **Máy:** Windows 11, AMD64 Family 25, Python 3.12.0. Base commit của đề: `b0ce8bf`.

## 09 · Limitations & ethics

- **PII / license:** dữ liệu synthetic CC0, không có người hay biển số thật. Không dùng `data/real/`.
- **Điều chưa biết:**
  - Số trên eval chưa có.
  - Dev chỉ có 4 ảnh đêm trong khi eval có 16, nên mọi kết luận về ảnh đêm đều yếu.
  - Ngưỡng của từng họ dựa trên 4–10 ảnh lỗi dev.
- **Nếu scale 10×:** tốc độ đủ (~3.5 ms/ảnh). Nhưng ngưỡng tuyệt đối (glare > 235, Laplacian 31.5) sẽ trôi khi đổi
  camera, độ phân giải hoặc mức nén gốc. Cần hiệu chỉnh lại theo từng camera và mỗi lô cần một heldout có nhãn.
- **Pháp lý:** đây là QC chất lượng ảnh, không phải đánh giá tuân thủ. Báo cáo không tuyên bố gì về GDPR hay luật
  Việt Nam.
- **Kế hoạch cải tiến:**
  1. Glare tương đối theo `light` (failure case 1).
  2. Ngưỡng blur và compression theo `camera_id`.
  3. Cho phép đa nhãn lỗi.
  4. Thu thêm ảnh đêm có nhãn trước khi tin vào slice night.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence:**

| Học viên | Git author | Module / vai trò | Commit SHA | Trình bày? |
|---|---|---|---|---|
| Pham Huynh (2A202603135) | phamhuynhmk1-png <phamhuynhmk1@gmail.com> | Toàn bộ: config, pipeline, baseline, tools, tests, report (làm cá nhân) | 5c1adcc (fix line endings), bce2d02 (pipeline + baseline + tests), 72c9f36 (reports) | Có |

## 11 · References

| Nguồn | Ngày xem | Ý nghĩa |
|---|---|---|
| `docs/tracks/T02.md` (handout) | 2026-11-14 | Đề bài, metric, baseline bắt buộc, 3 failure mode, gợi ý kết hợp đặc trưng và chuẩn hoá theo light/camera |
| ACDC: Adverse Conditions Dataset (ETH Zürich), https://acdc.vision.ee.ethz.ch/ | 2026-11-14 (theo Research Shelf của handout) | Ảnh đêm/sương là điều kiện hợp lệ cần giữ, nên độ tối phải so theo điều kiện |
| ISO/IEC TR 5259-6 (data quality for analytics/ML) | 2026-11-14 (theo Research Shelf của handout) | Khung quy trình đánh giá chất lượng dữ liệu, cần ghi provenance và quyết định có owner |

---

**Bảng 3 · Checklist chốt khi nộp**

- [x] Link repo và tag cuối cùng trong README và report; input mẫu không chứa PII thật.
- [x] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [x] Có baseline/PoC metrics (dev nội bộ; eval chờ giảng viên), ≥1 failure case, reviewer evidence, decision log.
- [x] Có license/source provenance, data splits, seed, offline environment, GT/eval N.
- [x] Mỗi thành viên có ≥1 commit thực chất.
