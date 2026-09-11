# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 468, "cab", 1, 0.510915, "ImageNet-1K".
- Record này mô tả toàn ảnh như thế nào? Record này cho biết mô hình nhận diện toàn bộ ảnh có khả năng cao nhất là lớp "cab".
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? Giữ cả 3 vì chúng phục vụ cho những việc khác nhau: ID là định danh duy nhất, tên lớp là nhãn dễ đọc với người dùng, tên taxonomy cho biết lớp ấy nằm ở hệ phân loại nào.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline phải nêu rõ gán nhãn cho ai, theo tiêu chí nào, ưu tiên ra sao và xử lý các trường hợp mơ hồ thế nào.
- Vì sao model score không phải ground truth? Model score chỉ phản ánh kết quả tính toán của checkpoint. Ground truth phải do con người xác nhận dựa trên taxonomy và guideline của dự án.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): "person", 0.769676, [0.08, 256.79, 18.39, 313.12], 18.32, 56.33.
- Diễn giải vị trí box bằng lời: Record này phát hiện người với độ tin cậy 76,9%. Phát hiện một phần người ở sát góc trái, nửa dưới ảnh; box rất hẹp nên chỉ bao phủ phần cơ thể bị khuất.
- So sánh số prediction ở hai threshold: Threshold 0.20 có 17 vật thể, threshold 0.35 có 11 vật thể. Suy ra ngưỡng càng cao, chỉ giữ các dự đoán tự tin hơn nên số vật thể phát hiện được giảm.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Threshold thấp tăng bao phủ nhưng tăng review; threshold cao giảm review nhưng có thể bỏ sót.
- Đề xuất một quy tắc box chặt: Chỉ giữ box rõ, ôm sát phần nhìn thấy và đủ score.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Object bị che/cắt mép, guideline cần quyết định có cần gắn nhãn phần nhìn thấy không, mức độ nhìn thấy tối thiểu và khi nào cần escalation.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): "traffic-001", "bus", 0.925745, [148.0, 189.0].
- Polygon bổ sung chi tiết gì so với box? Polygon mô tả đường viền chính xác của vật thể, box chỉ là khung chữ nhật bao quanh.
- `instance_id` dùng để làm gì và không phải loại ID nào? instance_id phân biệt từng cá thể cùng lớp; không phải class_id hay ID của taxonomy.
- Đề xuất một quy tắc biên mask: Chỉ tô phần vật thể nhìn thấy, bám sát biên rõ ràng.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Cần guideline xác định ranh giới, phần nào được tô và khi nào cần escalation.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | class_id, class_name, taxomony | nhiều chủ thể, nhãn mơ hồ | chọn nhãn theo guideline | đúng lớp, taxonomy |
| Phát hiện vật thể | object: class, score, bounding box | box lệch, object nhỏ/bị cắt/che | vẽ box sát phần nhìn thấy | đủ object, đúng lớp, box đúng |
| Instance segmentation | instance: class, instance_id, polygon | biên mờ, vật thể chạm/che nhau | Tô theo biên nhìn thấy | mask sát biên, tách đúng từng instance |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không để lộ họ tên, MSSV hoặc dữ liệu cá nhân trong báo cáo, output hay file ZIP.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Lab Coach

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
