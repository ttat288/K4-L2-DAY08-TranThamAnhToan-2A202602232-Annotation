# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Trần Thẩm Anh Toàn

Công cụ gán nhãn đã dùng: CVAT chạy bằng Docker trên máy cá nhân (bản 2.75.1, `http://localhost:8888`),
định dạng nhập/xuất Ultralytics YOLO Detection 1.0. Không sửa trực tiếp file nhãn `.txt`.

Mọi con số dưới đây lấy từ `reports/rounds_table.md`, `outputs/metrics_round*.json`,
`outputs/selection_round1.csv`, `outputs/selection_round2.csv` và `outputs/round1_diff.md`. Nhãn test
do mô hình tạo nên không được coi là chân lý tuyệt đối.

**Tóm tắt:** Một vòng học chủ động đã hoàn thành: 12 ảnh, sửa 169 box gợi ý thành 335 box. Sau fine-tune,
AP50 trên tập test **giảm từ 0.771 xuống 0.601 (−0.171)**. Mô hình mới gần như không nhận nhầm
(P = 1.000) nhưng bỏ sót rất nhiều (R = 0.117). Kết luận của mình: **chưa làm vòng 2 theo cấu hình hiện
tại**; cần kiểm tra cách fine-tune và nhãn tham chiếu trước.

## 1. Dữ liệu và cách chia tập

Video quay bằng camera cố định, 2.5 ảnh/giây, nên hai ảnh liền nhau (cách 0.4 s) gần như giống hệt, và
mỗi chiếc xe nằm trong khung hình vài giây. Vì thế dữ liệu được chia theo trục thời gian
(`data/DATA.md`): 20 ảnh test ở 4 đoạn quanh giây 20, 60, 100, 140; 112 ảnh vùng đệm bị loại; 268 ảnh
còn lại làm pool. Ảnh pool gần ảnh test nhất vẫn cách 4.4 s.

Nếu chia ngẫu nhiên, một ảnh test gần như chắc chắn có "anh em sinh đôi" cách nó 0.4 s nằm trong tập
huấn luyện, chứa đúng những chiếc xe đó ở gần như cùng vị trí. Mô hình sẽ được chấm trên xe nó đã học
thuộc, nên **số đo trên tập test bị lệch lên (cao hơn thực tế)**. Đây là rò rỉ dữ liệu (data leakage):
điểm cao nhưng không cho biết mô hình xử lý cảnh mới tốt đến đâu. Vùng đệm 4 s bảo đảm xe trong test
đã rời khung hình trước khi ảnh pool gần nhất xuất hiện.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 trong `rounds_table.md` (20 ảnh test, 403 box tham chiếu, bỏ qua 14 box cao dưới 16 px):

| vòng | model | ảnh train | box train | AP50 | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Theo `metrics_round0.json`: 197 phát hiện đúng (TP), 16 nhận nhầm (FP), 206 bỏ sót (FN). **Lỗi chính
là bỏ sót**, không phải nhận nhầm: mô hình tìm được chưa tới một nửa số xe.

Quan sát trên `outputs/compare_round0.jpg` (box vàng là xe bị bỏ sót, đỏ là nhận nhầm):

- **Xe nhỏ ở xa, gần đường chân trời:** nhiều box vàng dồn ở cụm xe phía trên bên trái của
  `frame_0050`, `frame_0150`, `frame_0350`. Khớp với `R small = 0.182` (chỉ 12/66 xe nhỏ được tìm thấy).
- **Xe chỉ thấy cụm đèn hậu đỏ, đi xa dần ở làn bên phải:** ở `frame_0150`, cụm đèn đỏ bên phải bị vẽ
  thành box lệch hoặc chồng lên nhau (box đỏ và vàng lẫn nhau), thay vì 2 box riêng như tham chiếu.
- **Xe gần, sáng đèn pha, ở mép dưới ảnh:** ví dụ xe dưới cùng bên trái của `frame_0050` và
  `frame_0250` vẫn là box vàng dù nhìn rất rõ. Vì vậy `R large` chỉ đạt 0.561, gần bằng `R medium`.
  Mô hình không có box nào khớp IoU ≥ 0.5 với các xe này; khả năng là do xe bị cắt ở mép
  ảnh và vùng lóa đèn pha lớn làm khó xác định thân xe. Cần xem ảnh cỡ lớn để khẳng định.
- **Nhận nhầm:** ở `frame_0350`, một box đỏ lớn ở phía trái bao trùm vùng lóa đèn cạnh các xe, không khớp xe nào.

Độ phủ theo kích thước cho thấy mô hình COCO chưa quen cảnh đêm: yếu nhất với xe nhỏ và xe chỉ còn
thấy đèn, và ngay cả xe lớn cũng chỉ tìm được khoảng một nửa.

**Trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai:** trong
`frame_0350`, nhãn tham chiếu có một box nhỏ ở **góc trên bên trái, ngang đường chân trời**. Vị trí này
nằm trên dải đèn thành phố/biển hiệu phát sáng, không phải mặt đường. Theo `GUIDELINE_LABEL.md`, biển
báo phát sáng không được gán nhãn. Tương tự, `frame_0250` có một box tham chiếu hẹp ở **sát mép phải ảnh**
chỉ chứa một vệt sáng. Nếu hai box này sai, mô hình bị tính thêm FN oan. Mình không sửa `data/test/labels/`,
chỉ ghi nhận ở đây.

## 3. Chiến lược chọn mẫu

**Công thức bằng lời:** `score = 0.5·U + 0.3·A + 0.2·D`. Một ảnh được ưu tiên khi:

- `U` (độ bất định, trọng số 0.5) cao: lấy 5 box khó nhất trong ảnh, mỗi box có độ bất định
  `u = 1 − |2·conf − 1|`, cao nhất khi độ tin cậy bằng 0.5, tức mô hình "năm ăn năm thua";
- `A` (số box mơ hồ, trọng số 0.3) lớn: đếm box có độ tin cậy 0.15–0.50, chia cho giá trị lớn nhất pool;
- `D` (độ đa dạng thời gian, trọng số 0.2) lớn: ảnh càng xa ảnh đã gán nhãn (tối đa 10 s) càng có lợi.
  Ở vòng 1 chưa có ảnh nào được gán nhãn nên mọi ảnh có `D = 1.0`.

**`MIN_GAP_S = 2.0`:** khi chọn lô từ trên xuống, hai ảnh trong cùng lô phải cách nhau ít nhất 2 giây,
để tránh tốn công gán nhãn cho các ảnh gần trùng. Vì vậy lô 12 ảnh không phải 12 dòng đầu CSV:
`frame_0372` (hạng 6), `frame_0368` (hạng 9), `frame_0330` (hạng 12) bị loại vì cách ảnh đã chọn dưới 2 s.

**Dẫn chứng từ `reports/SELECTION.md`:**

- `frame_0182` (hạng 1, score 0.9591, `A = 1.0`): điểm cao phản ánh đúng chỗ yếu, từ 13 box gợi ý phải
  thêm 14 xe.
- `frame_0331` (hạng 5, score 0.9154, 47 box, 18 mơ hồ): ảnh có nhiều box nhận nhầm nhất lô (xóa 5), có box
  gộp 2 xe. Công sửa cao nhưng đáng làm.
- `frame_0326` (hạng 4, score 0.9155): được chọn dù chỉ cách `frame_0331` đúng 2.0 s. Ảnh ghép cho thấy gần
  như cùng dòng xe. Với ngân sách 5 ảnh, mình bỏ ảnh này để giữ `frame_0331`.
- Ảnh khác: `frame_0372` (hạng 6, score 0.9101) có điểm cao hơn 7 ảnh được chọn nhưng chỉ cách
  `frame_0369` 1.2 s; loại là hợp lý vì gán nhãn thêm gần như không mang thông tin mới.

Mình cân nhắc thêm **công gán nhãn**: các ảnh điểm cao đều là cảnh dày xe; trung bình mỗi ảnh phải thêm
khoảng 15 box (183/12), riêng `frame_0369` phải thêm 24 box.

**Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?** Không. Điểm cao chỉ nói mô hình
đang phân vân ở ảnh đó. Nó không tính đến: (1) lỗi "tự tin sai" (box gộp 2 xe có độ tin cậy cao vẫn không
làm tăng điểm); (2) nhãn ảnh đó có gán nhất quán được không; (3) ảnh đó có giống tập test không. Kết quả
vòng 1 là bằng chứng trực tiếp: chọn đúng ảnh điểm cao nhưng AP50 vẫn giảm (mục 4).

## 4. Các vòng học chủ động (active learning)

Bảng từ `reports/rounds_table.md` (ngưỡng IoU 0.5; P, R, F1 tính tại conf 0.25):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 335 | 0.601 | -0.171 | 1.000 | 0.117 | 0.209 | 0.000 | 0.098 | 0.439 |

### 4.1. Mức độ sửa nhãn gợi ý (`outputs/round1_diff.md`)

| Model đề xuất | Sau khi sửa | Giữ nguyên (accepted) | Sửa khung (edited) | Xóa (FP của model) | Thêm (FN của model) | Tỉ lệ giữ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 169 | 335 | 133 | 19 | 17 | 183 | 79% |

Số box sau sửa gần gấp đôi: lỗi của nhãn gợi ý chủ yếu là **bỏ sót** (183 box thêm so với 17 box xóa).
Điều này khớp với R@0.25 = 0.489 của cold start ở mục 2. Ảnh bị thiếu nhiều nhất là `frame_0369` (+24), ảnh
nhận nhầm nhiều nhất là `frame_0331` (5 box bị xóa).

### 4.2. Thay đổi số đo

- **AP50:** 0.771 → 0.601, **−0.171 so với cold start** (vòng trước của vòng 1 chính là cold start, nên mức
  giảm so với vòng trước cũng là −0.171). Mức giảm này lớn hơn nhiều so với ngưỡng nhiễu khoảng 0.01 mà
  `DATA.md` nêu cho tập 20 ảnh, nên đây là thay đổi thật, không phải dao động ngẫu nhiên.
- **Tốt lên:** độ chính xác P@0.25 từ 0.925 lên 1.000; FP từ 16 xuống 0 (`metrics_round1.json`).
- **Xấu đi:** độ phủ ở mọi nhóm kích thước. Xe nhỏ 0.182 → 0.000, xe vừa 0.547 → 0.098, xe lớn 0.561 → 0.439.
  TP giảm từ 197 xuống 47, FN tăng từ 206 lên 356. Xe lớn ở gần giảm ít nhất.

### 4.3. Một ca cụ thể trên `outputs/compare_round1.jpg`

Ở `frame_0050`: cold start có TP 11, FP 2, FN 7; vòng 1 có TP 3, FP 0, FN 15. Hai box đỏ nhận nhầm của cold
start ở cụm xe xa phía trên đã biến mất (tốt hơn), nhưng hầu hết xe cỡ vừa ở giữa đường, vốn được cold
start tìm đúng (box xanh), nay thành box vàng bị bỏ sót. Chỉ còn các xe lớn sát camera được giữ lại.
`frame_0150`, `frame_0250`, `frame_0350` có cùng xu hướng (FP về 0, TP còn 2–3).

**Lý do có thể kiểm:** mô hình vòng 1 cho độ tin cậy rất thấp. Ở ngưỡng 0.25 nó chỉ giữ box nó rất chắc
chắn, nên P = 1.0 nhưng R = 0.117. Bằng chứng thứ hai trong `outputs/selection_round2.csv`: khi mô hình vòng 1
dự đoán trên pool (ngưỡng 0.01), các ảnh dẫn đầu chỉ có 1–8 box (ví dụ `frame_0023` có 1 box, `frame_0009`
có 2 box), trong khi 12 ảnh mình gán nhãn có 22–37 xe mỗi ảnh. Nguyên nhân khả dĩ nhất là **cách fine-tune**: mô hình
được huấn luyện lại từ `yolov8n.pt` (80 lớp COCO) sang 1 lớp `car`, nên phần phân lớp của đầu dự đoán
phải học lại từ đầu chỉ với 12 ảnh, 50 epoch và không có tập kiểm định. Mô hình mất lợi thế COCO mà chưa
đủ dữ liệu để thay thế. Kiểm tra được bằng cách vẽ phân bố độ tin cậy của mô hình vòng 1 trên ảnh test.

### 4.4. Phân biệt ba nguồn bằng chứng

| Nguồn | Nội dung | Điều nó nói lên |
| --- | --- | --- |
| Quan sát độc lập (`BLIND_SCAN.md`, đã khóa trước khi xem nhãn AI) | `frame_0270`: tự đếm 26 xe; dự đoán AI dễ nhận nhầm cây đèn đỏ trên dải phân cách và bỏ sót xe xa cuối đường chỉ thấy đèn. | Giả thuyết của người, trước khi bị nhãn AI ảnh hưởng. |
| Lỗi nhãn gợi ý đã sửa (`REVIEW_LOG.csv`, `round1_diff.md`) | `frame_0270`: AI chỉ gợi ý 13 box; sau khi sửa còn 25 box (giữ 12, sửa 1, **xóa 0**, thêm 12). Cả lô: 6 box AI gộp 2 xe làm một (vd. `frame_0227`, `frame_0312`), nhiều box vẽ rộng hơn thân xe (vd. `frame_0392` thu còn 53% diện tích). | Giả thuyết "bỏ sót xe" **đúng** (thiếu gần một nửa). Giả thuyết "nhận nhầm đèn đỏ" **không xảy ra** ở ảnh này (0 box bị xóa). Lỗi thực tế thường gặp hơn là box gộp 2 xe và box quá rộng. Chênh 26 (tự đếm) và 25 (nhãn cuối) có thể do một xe rất xa dưới 16 px, loại được phép bỏ qua. |
| Kết quả mô hình sau train (`metrics_round1.json`, `compare_round1.jpg`) | AP50 giảm 0.171, R giảm mạnh. | Nhãn tốt hơn **không tự động** cho mô hình tốt hơn; phải xét cả cách huấn luyện và bộ tham chiếu. |

### 4.5. Một ca khó theo guideline

**Hai xe đi sát nhau, AI gộp làm một box** (`frame_0227`, xe ở làn bên trái giữa ảnh). Mình áp dụng dòng
"Hai xe đứng sát nhau: vẽ hai box riêng, không gộp làm một" của `GUIDELINE_LABEL.md`: xóa box gộp, vẽ lại
2 box ôm sát từng thân xe (phần nhìn thấy nếu bị che), không tính vệt đèn pha trên mặt đường. Mình giữ
cách xử lý này cho cả 6 trường hợp gộp trong lô để nhãn nhất quán.

Với **xe ở xa chỉ thấy đèn**, mình vẽ theo thân xe đoán được quanh cụm đèn; xe cao dưới khoảng 16 px thì
không bắt buộc vì không được tính điểm.

## 5. Kết luận và giới hạn

**So với cold start:** vòng 1 kém hơn (AP50 −0.171, R 0.489 → 0.117), chỉ tốt hơn ở độ chính xác (FP 16 → 0).

**Quyết định: dừng, chưa làm vòng 2 với cấu hình hiện tại.** Lý do: mức giảm lớn cho thấy vấn đề nằm ở
cách fine-tune hoặc bộ tham chiếu, không phải ở việc thiếu thêm 12 ảnh. Làm thêm vòng với cùng cấu hình
nhiều khả năng lặp lại kết quả. Ngoài ra, lô vòng 2 do chính mô hình yếu này chọn và gợi ý nhãn, nên nhãn
gợi ý chỉ còn 1–8 box/ảnh, tức gần như phải vẽ tay lại toàn bộ.

**Hai ca còn yếu/bất định đề xuất cho vòng sau:**

1. **Xe nhỏ ở xa gần đường chân trời** (R small = 0.000 ở vòng 1, 0.182 ở vòng 0). Chi phí gán nhãn cao: mỗi
   ảnh có hàng chục xe nhỏ san sát, nhiều xe dưới 16 px không được tính điểm, nên cần quy ước rõ trước
   khi gán. Nguy cơ gần trùng: xe xa di chuyển chậm trên ảnh, các ảnh cách vài giây gần như giống nhau.
2. **Cụm đèn hậu đỏ và xe đi sát nhau ở làn bên phải** (vd. `frame_0150` ở tập test, các ca gộp 2 xe trong lô 1).
   Chi phí vừa phải: chủ yếu tách box, chỉnh khung. Nguy cơ gần trùng: `selection_round2.csv` đã có dấu
   hiệu, `frame_0384` (153.6 s) chỉ cách `frame_0380` đã gán nhãn 1.6 s (`D = 0.16`), và 4 ảnh
   `frame_0284`–`frame_0304` dồn trong 8 giây (113.6–121.6 s).

Nếu làm vòng 2, mình sẽ tự chọn lại theo hai ca trên và dùng mô hình cold start để gợi ý nhãn, thay vì
dùng nguyên lô `to_label/round2/`.

**Giới hạn ảnh hưởng đến kết luận:**

- **Tập test chỉ 20 ảnh** (4 đoạn thời gian): một vài ảnh khó có thể kéo AP50 lên xuống. Mức −0.171 đủ lớn
  để tin là giảm thật, nhưng không đủ để khẳng định tỉ lệ chính xác cho mọi cảnh đêm.
- **Luật bỏ qua xe dưới 16 px** (14/417 box): các xe xa nhất không được tính, nên số đo đánh giá thấp mức
  khó thật của bài toán xe xa.
- **Nhãn tham chiếu do mô hình tạo, chưa được người rà:** AP50 chỉ đo mức khớp với một mô hình khác. Nếu
  mô hình tham chiếu vẽ box theo phong cách khác nhãn của mình (ôm cả vùng lóa, gộp xe, hoặc box ở biển
  hiệu như `frame_0350`), fine-tune theo guideline có thể làm AP50 **giảm dù nhãn đúng hơn**. Vì vậy mình
  không kết luận mô hình vòng 1 kém hơn ngoài thực tế chỉ dựa trên AP50.

**Nếu AP50 giảm, kiểm tra gì trước khi train thêm:**

1. **Nhãn huấn luyện:** mở `labels/round1/` trên ảnh để chắc không lệch tọa độ khi xuất từ CVAT (đã
   qua `pack_labels.py`, mã lớp đều là 0, không có ảnh test trong lô).
2. **Độ tin cậy của mô hình:** vẽ phân bố conf trên ảnh test. Nếu phần lớn box đúng nằm dưới 0.25, vấn đề
   là hiệu chỉnh độ tin cậy, không phải mô hình không thấy xe.
3. **Cách fine-tune:** thử giữ đầu dự đoán COCO (fine-tune từ lớp `car/bus/truck` thay vì khởi tạo 1 lớp
   mới), tăng số ảnh hoặc số epoch, hoặc tách vài ảnh pool làm tập kiểm định để chọn checkpoint.
4. **Nhãn tham chiếu:** so trực quan phong cách box tham chiếu với guideline trên vài ảnh test; ghi các box
   nghi sai (như `frame_0350`, `frame_0250`) để người phụ trách rà.
5. **So với đối chứng ngẫu nhiên:** chạy lại với `STRATEGY = "random"` để biết mức giảm do cách chọn mẫu hay
   do cách huấn luyện.

**Tự QC:** `BLIND_SCAN.md` được khóa trước khi mở nhãn AI và không sửa sau đó; `REVIEW_LOG.csv` đã được đối
chiếu với so sánh nhãn gợi ý và nhãn đã sửa (các box gộp 2 xe và box bị thu nhỏ đều khớp với
`round1_diff.json`); `check_submission.py` được chạy trước khi nộp.
