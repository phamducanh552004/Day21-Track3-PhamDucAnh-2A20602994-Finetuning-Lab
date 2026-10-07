# Reflection — Lab 21

Phạm Đức Anh — 2A202602994.

1. Ngạc nhiên nhất: target 0.970 nhưng verdict FAILED vì regression giảm 0.380. Một điểm task cao không đủ bảo đảm chất lượng tổng thể.
2. NB4 lâu nhất: 1273/2815 giây, gần một nửa pipeline vì có ba lượt train đối chứng.
3. Tôi không còn xem train loss thấp là bằng chứng cấu hình tốt nhất: attention-only loss 0.5364 nhưng hoà target với correct.
4. AI assistant thao tác Colab, theo dõi, thu thập kết quả và soạn báo cáo. Cách gọi target 0.970 là “97% task” trước đó dễ bị hiểu thành tỷ lệ ticket hoàn toàn đúng; báo cáo đã phân biệt field accuracy và exact match. Chạy xong pipeline chưa đồng nghĩa nộp được vì REPORT ban đầu vẫn còn placeholder.
5. Với khách hàng thật, đầu tiên thống nhất benchmark và ngưỡng trước khi train. Chạy strong-prompt baseline và regression để cân nhắc lợi ích so với chi phí vận hành.
