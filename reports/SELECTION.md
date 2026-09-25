# Vì sao chọn lô này?

Nguồn số liệu: `outputs/selection_round1.csv` (268 ảnh pool được xếp hạng bằng mô hình cold start),
`outputs/selection_round1.jpg` (ảnh ghép 12 ảnh được chọn), `to_label/round1/batch.json`
(`MIN_GAP_S = 2.0`, `AL_K = 12`). Ở vòng này chưa có ảnh nào được gán nhãn nên mọi ảnh đều có
`D = 1.0`; thứ hạng chỉ do `U` (độ bất định) và `A` (số box mơ hồ) quyết định. Cột `empty` bằng
`False` với cả 268 ảnh: không có ảnh nào mô hình không dự đoán được box, nên `EMPTY_BONUS` không tác
động ở vòng này. Vì vậy, tiêu chí phụ mình dùng là **ảnh gần trùng theo thời gian**.

## 1. Nếu chỉ có ngân sách rà 5 ảnh

Mình xét 50 dòng đứng đầu CSV và chọn theo nguyên tắc: điểm cao trước, nhưng mỗi ảnh phải đại diện
cho một đoạn thời gian khác nhau (camera đứng yên, ảnh cách nhau dưới vài giây gần như cùng một dòng
xe).

| Thứ tự rà | Frame | Hạng CSV | Score | t (giây) | U / A | Lý do |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | `frame_0182.jpg` | 1 | 0.9591 | 72.8 | 0.918 / 1.000 | Điểm cao nhất; 18 box mơ hồ trên 28 box, đoạn giữa video. |
| 2 | `frame_0331.jpg` | 5 | 0.9154 | 132.4 | 0.831 / 1.000 | Nhiều box nhất trong top 5 (47 box, 18 mơ hồ): cảnh dày xe, nhiều chỗ mô hình phân vân. Chọn ảnh này **thay cho** `frame_0326` (hạng 4, 130.4 s) vì hai ảnh chỉ cách nhau 2.0 s, gần trùng cảnh. |
| 3 | `frame_0369.jpg` | 2 | 0.9324 | 147.6 | 0.932 / 0.889 | Đại diện đoạn cuối video. Bỏ qua `frame_0380` (hạng 3, 152.0 s) vì chỉ cách 4.4 s, cùng đợt xe; với ngân sách 5 ảnh, ưu tiên đoạn thời gian khác. |
| 4 | `frame_0099.jpg` | 8 | 0.9063 | 39.6 | 0.946 / 0.778 | `U` cao nhất trong top 10 và là ảnh có score cao nhất ở đoạn đầu video (dưới 70 s). |
| 5 | `frame_0270.jpg` | 13 | 0.8878 | 108.0 | 0.909 / 0.778 | Lấp khoảng trống 72.8–132.4 s; có xe tải lớn ở làn giữa (xem ảnh ghép), khác kiểu xe so với các ảnh trên. |

Năm ảnh này trải từ giây 39.6 đến 147.6, hai ảnh gần nhau nhất vẫn cách 15.2 giây (`frame_0331` và `frame_0369`). Đổi lại, mình chấp
nhận bỏ hai ảnh có điểm cao hơn (`frame_0380`, `frame_0326`) vì chúng lặp lại cảnh của ảnh đã chọn.

## 2. Ba frame trong lô 12 ảnh mô hình đã chọn

- **`frame_0182.jpg`** (hạng 1, score 0.9591, t = 72.8 s): `A = 1.0` là giá trị lớn nhất pool (18 box
  có độ tin cậy 0.15–0.50). Sau khi rà, ảnh này đúng là nhiều lỗi: từ 13 box gợi ý thành 26 box, phải
  thêm 14 xe bị bỏ sót (`outputs/round1_diff.md`). Điểm cao phản ánh đúng chỗ mô hình yếu.
- **`frame_0331.jpg`** (hạng 5, score 0.9154, t = 132.4 s): 47 box dự đoán, 18 box mơ hồ. Đây là ảnh
  có nhiều box nhận nhầm nhất trong lô: xóa 5 box, sửa 3 box, thêm 19 box. Trong đó có box AI gộp 2 xe
  đứng sát nhau thành một khung.
- **`frame_0326.jpg`** (hạng 4, score 0.9155, t = 130.4 s): được chọn cùng lô với `frame_0331` dù chỉ
  cách 2.0 s, đúng bằng `MIN_GAP_S`. Ảnh ghép cho thấy hai cảnh gần như cùng một dòng xe. Đây là minh
  chứng `MIN_GAP_S = 2.0` chỉ chặn ảnh gần trùng ở mức tối thiểu; hai ảnh vẫn mang nhiều thông tin
  trùng lặp. Tương tự, cặp `frame_0182`/`frame_0187` (72.8 s và 74.8 s) cũng cách nhau đúng 2.0 s.

Ngược lại, `frame_0372` (hạng 6), `frame_0368` (hạng 9) và `frame_0330` (hạng 12) có điểm cao nhưng
`selected = False`, vì chúng cách ảnh đã chọn dưới 2 giây (`frame_0369`, `frame_0331`).

## 3. Một frame điểm cao nhưng không nên chọn

**`frame_0372.jpg`** (hạng 6, score 0.9101, t = 148.8 s) có điểm cao hơn 7 ảnh đã được chọn, nhưng chỉ
cách `frame_0369` (147.6 s) 1.2 giây. Với camera cố định và 2.5 ảnh/giây, hai ảnh này gần như cùng các
chiếc xe ở cùng vị trí; gán nhãn cả hai tốn gấp đôi công mà mô hình học được rất ít điều mới. Công cụ
đã loại đúng ảnh này nhờ `MIN_GAP_S`.

Ghi chú thêm (không bắt buộc): ảnh điểm thấp nhất `frame_0195.jpg` (hạng 268, score 0.5721, t = 78.0 s)
vẫn có 20 box dự đoán. Nên rà ngẫu nhiên một vài ảnh điểm thấp như vậy để kiểm tra các lỗi mà mô hình
"tự tin sai" (tin cậy cao nhưng sai), vì độ bất định không bắt được loại lỗi này.

## 4. Điều phép chọn này chưa chứng minh

- Điểm bất định chỉ cho biết **mô hình đang phân vân**, không chứng minh ảnh đó sẽ giúp mô hình tốt
  lên. Thực tế vòng 1: 12 ảnh điểm cao được chọn và gán nhãn kỹ (335 box) nhưng AP50 trên tập test
  **giảm** từ 0.771 xuống 0.601 (`reports/rounds_table.md`).
- `U` và `A` được tính từ độ tin cậy của chính mô hình cold start. Lỗi "tự tin sai" (ví dụ box gộp 2
  xe có độ tin cậy cao) không làm tăng điểm, nên phép chọn có thể bỏ qua chúng.
- Phép chọn không đo **công gán nhãn**. Các ảnh điểm cao đều là cảnh dày xe (28–47 box dự đoán mỗi ảnh), mỗi ảnh
  trung bình phải thêm khoảng 15 box (183 box thêm / 12 ảnh).
- `D = 1.0` cho mọi ảnh ở vòng đầu, nên độ đa dạng thời gian chưa đóng vai trò gì ngoài `MIN_GAP_S`.
