# Quét độc lập trước khi xem pre-label

Frame: frame_0270.jpg

Số xe nhìn thấy bằng mắt: 26 xe từ 4 bánh trở lên

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:

1. Dải phân cách có cây đèn màu đỏ, trông khá giống đèn hậu (đuôi) xe nhưng không phải xe; AI dễ nhận nhầm thành xe (False Positive).
2. Xe ở xa cuối đường, nhìn không rõ thân xe, chỉ đoán được qua đèn; AI dễ bỏ sót hoặc vẽ khung sai (False Negative / khung lệch).

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
