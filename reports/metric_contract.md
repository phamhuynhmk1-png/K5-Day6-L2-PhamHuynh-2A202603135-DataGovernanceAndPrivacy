# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T02 · Raw QC Triage |
| Primary metric (tên đúng như evaluator in ra) | `Defect Recall@B` (kèm `Precision@B`) |
| Tăng tốt hay giảm tốt? | higher |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | image |
| Công thức + numerator / denominator | Recall@B = #ảnh lỗi thật (GT `is_defect=1`) nằm trong top-B theo `score` / #ảnh lỗi thật trong split eval. Precision@B = cùng numerator / B |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | theo `image_id`; sắp `score` giảm dần, hoà điểm giữ thứ tự file (mergesort) |
| Candidate pool cố định | 40 ảnh split `eval` (10 cảnh gốc × 4), cùng pool cho baseline và PoC; pipeline chấm điểm cả 96 ảnh, evaluator lọc về eval |
| Review budget B / K | B = 20 |
| Seed | 42 (pipeline không có bước ngẫu nhiên → chạy 2 lần ra file giống hệt, có test) |
| Threshold policy (chọn trên dev như thế nào) | Nhãn tay 56 ảnh dev (`configs/t02_dev_manual_labels.csv`, chỉ nhìn ảnh dev). `tools/calibrate_t02.py`: mỗi họ lỗi (blur/dark/glare/compression) ngưỡng = điểm giữa (thang log) giữa ảnh sạch gần nhất và ảnh lỗi gần nhất trên dev; độ sáng tham chiếu theo `light` = median ảnh sạch dev. Baseline: 1 ngưỡng Laplacian tối đa F1 trên dev. Chốt vào `configs/lab.yaml` |
| Eval split (không dùng để tune) | `eval` (40 ảnh). Không có nhãn eval trong repo học viên; nhóm chỉ xem ảnh eval **sau khi** chốt ngưỡng, để phân tích failure case, không sửa ngưỡng |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | miss (ảnh lỗi nằm ngoài top-B → bị đưa vào training) là lỗi xấu nhất; false alarm trong top-B tốn công reviewer và có thể loại nhầm ảnh đêm hữu ích |
| Slices sẽ báo cáo (≥2) | `light` ∈ {day, dusk, night} (false alarm + recall trong top-B); `defect_type` ∈ {blur, dark, glare, compression}; ngoài ra `camera_id` (cam_B quang học mềm) trong phân tích lỗi |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | Wilson 95% CI cho Recall@B và Precision@B (N lỗi eval nhỏ, ~20); báo kèm numerator/denominator |
| Ngày giờ chốt + người chốt | 2026-11-14 · Pham Huynh (Data QC owner) |

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện.
