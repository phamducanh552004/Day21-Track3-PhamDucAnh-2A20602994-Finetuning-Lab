# Lab 21 — Evaluation Report

**Họ tên:** Phạm Đức Anh — **MSSV:** 2A202602994 — **Ngày:** 08/10/2026.

**Tier:** T4; **model:** `unsloth/Qwen3.5-4B`; **GPU:** Tesla T4, 14.6 GiB thực tế. FP16 vì T4 không hỗ trợ BF16. Mã nguồn thực nghiệm: `d27c1c02ebe99f32f52f706be88b4c30fb1d7fca`.
[Notebook Colab](https://colab.research.google.com/drive/1ChKb3kpcQM96O7MZUjg5hvgWp5OsJdwJ).

## 1. Setup

Chọn 250 ticket CSKH tiếng Việt mặc định, đầu ra JSON gồm intent, urgency, product, sentiment. Đây là tác vụ hẹp có bốn trường chấm được để so sánh prompt engineering và LoRA. Model 4B theo tier T4 cho phép chạy cả FP16 và QLoRA trên cùng GPU.

| Thuộc tính | Giá trị |
|---|---|
| Train / validation | 225 / 25, seed 42 |
| Evaluation | 50 target, 15 regression; EVAL_LIMIT bỏ trống |
| Mask | assistant-only |
| Token length | mean 93.1, p95 98, max 101 |
| max_length | 1024 |
| Epochs / optimizer steps | 2 / 30, giống nhau ở bốn run |
| Batch / accumulation | 1 / 16, effective batch 16 |
| Correct LoRA | text-linear, 12 modules, rank 16, alpha 32, LR 0.0001 |

Giữ max_length 1024 theo cấu hình tier đã dùng từ đầu, dùng chung cho tất cả đối chứng. Đây là giá trị dư so với p95; 256 phù hợp hơn cho một lần tối ưu sau. Mẫu dài nhất chỉ 101 token nên không bị cắt. Không tuyên bố 1024 tối ưu về tốc độ hoặc bộ nhớ.

## 2. Mask proof — CP1

`results/mask_proof.json`: 39/94 token được giám sát, tỷ lệ 0.4149; answer_is_supervised=true, question_is_masked=true. Preview phần tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

System prompt và câu hỏi được che bằng nhãn -100. Trên toàn train, 9014/20951 token được giám sát, 43.0%. `template_check.json` giữ thẻ mở và nội dung think trong ví dụ kiểm tra. Đây là kiểm tra cấu trúc template, không chứng minh model học suy luận: tập ticket chỉ có JSON nhãn, không có reasoning trace chứa nội dung.

## 3. Baseline đóng băng — CP2

NB2 hoàn thành trước NB3. Giữ nguyên evaluation và OPTIMIZED_PROMPT, hash `719e74d3b6232053`; smoke_mode=false, eval_limit=null. Không sửa ngưỡng sau khi thấy kết quả.

| Run | Target field accuracy | Regression | JSON format | Latency ms/item |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0.000 | 0.7911 | 0.000 | 3163.6 |
| (b) Base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1039.3 |
| (c) LoRA correct | 0.970 | 0.4111 | 1.000 | 1452.7 |

Prompt B mạnh hơn A theo target và format. Không sửa prompt gốc. Target 0.970 là điểm trung bình trên bốn trường, không phải 97% ticket hoàn toàn đúng. Có 44/50 ticket đúng cả bốn trường, 6 ticket đạt 0.75: 194/200 trường đúng.

## 4. Đối chứng — CP3/CP4

| Run | Placement | Rank | Trainable | LR | Train loss | Target | Train s | Peak VRAM GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correct | text-linear | 16 | 32464896 | 0.0001 | 0.6265 | 0.970 | 420.9 | 8.78 |
| attn_only | q,v | 283 | 32456704 | 0.0001 | 0.5364 | 0.970 | 267.2 | 8.79 |
| wrong_lr | text-linear | 16 | 32464896 | 0.00001 | 1.5702 | 0.000 | 389.5 | 8.78 |
| qlora | text-linear, base 4-bit | 16 | 32464896 | 0.0001 | 0.7058 | 0.940 | 458.0 | 3.86 |

**4.1 Rank và placement.** Attention-only rank 283 khớp ngân sách tham số với correct; chênh 8192, khoảng 0.0252%, nhỏ hơn ngưỡng 5%. Hai run hoà target dù attention-only có train loss thấp hơn. Không thể kết luận attention-only thua hoặc loss thấp hơn đồng nghĩa chất lượng cao hơn. Placement hẹp với rank được khớp vẫn đủ cho tác vụ JSON này; kết quả không chứng minh mọi tác vụ đều như vậy.

**4.2 Learning rate.** Wrong_lr chỉ giảm LR từ 1e-4 xuống 1e-5, giữ 30 bước. Loss giảm chậm, trung bình 1.5702 thay vì 0.6265. Target và JSON format bằng 0; ngân sách ngắn với LR này chưa học được hành vi JSON. Chỉ nhìn loss đang giảm sẽ khiến tôi kết luận sai rằng run đã dùng được.

**4.3 QLoRA.** VRAM giảm 8.78 xuống 3.86 GiB, tiết kiệm 4.92 GiB, khoảng 56.0%. Target giảm 0.03, train tăng 37.1 giây, latency tăng 1452.7 lên 1786.9 ms/item. Adapter QLoRA được đánh giá trên base 4-bit đúng với lúc train. Số đo ủng hộ việc có đánh đổi, không đủ để cấm QLoRA cho mọi trường hợp; nếu VRAM là giới hạn chính nó vẫn đáng cân nhắc. Các đối chứng chỉ đo target/format/latency, không đo regression riêng.

## 5. Phán quyết — CP5

**FAILED**, target delta +0.205, regression delta -0.380, vượt mức giảm tối đa 0.020. JSON format 1.000, valid_trace_rate=0.0.

Fine-tune cải thiện rõ tác vụ ticket nhưng không vượt toàn bộ cổng chất lượng. Regression giảm từ 0.7911 xuống 0.4111 cho thấy model mất một phần khả năng đáp ứng ngoài miền. Kết quả phù hợp với hiện tượng quên sau huấn luyện tập hẹp, nhưng đây là diễn giải từ số đo, không phải chứng minh cơ chế bên trong model. Latency tăng khoảng 39.8% so với prompt B dù không dùng system prompt dài. Tôi giữ nguyên verdict và tolerance thay vì làm yếu baseline hoặc đổi tập evaluation. Tập 15 câu regression còn nhỏ, cần benchmark lớn hơn trước quyết định triển khai. Hiện tại nên dùng base và prompt B cho hệ thống dùng chung. Nếu cần adapter chuyên biệt, thử thêm 1–5% replay data từ nguồn train độc lập, không sao chép câu evaluation vào train, rồi đánh giá lại. Trace rate 0 chưa đủ chứng minh reasoning collapse do dữ liệu không có trace huấn luyện; không nhận bonus này.

## 6. Định tính

Phép chạy bổ sung dùng cùng model, prompt và dữ liệu, lưu câu trả lời đầy đủ để kiểm tra định tính; không thay baseline đóng băng hoặc verdict. `results/qualitative_selected.json` chứa nguyên văn năm cặp được chọn. Artifact gốc `qualitative.json` cắt preview 70/90 ký tự; dấu cắt không có nghĩa output thực tế sai JSON.

| Nhóm / index | Yêu cầu rút gọn | Nhãn / yêu cầu đúng | Base B | Fine-tune | Nhận xét |
|---|---|---|---|---|---|
| target / 0 | Chuột không dây, “Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.” | doi_tra, cao, chuột không dây, tich_cuc | intent hoan_tien; 0.75 | Đúng 4 trường; 1.0 | FT thắng, phân biệt trả hàng và hoàn tiền |
| target / 1 | Ốp điện thoại, “Hoàn tiền. Sớm nhé. Bực mình.” | hoan_tien, trung_binh, ốp lưng điện thoại, tieu_cuc | urgency cao; 0.75 | Đúng 4 trường; 1.0 | FT thắng, urgency theo nhãn dataset |
| regression / 2 | 1 km bằng bao nhiêu mét? | 1000 | Trả lời 1000; 1.0 | JSON hoi_thong_tin, không có 1000; 0.0 | FT thua, áp format ticket lên câu kiến thức |
| regression / 3 | Viết câu chúc mừng sinh nhật bằng tiếng Việt | Câu chúc có “sinh nhật” | Câu chúc tự nhiên; 1.0 | JSON intent chuc_mung_sinh_nhat, không có cụm yêu cầu; 0.0 | FT thua, không thực hiện yêu cầu viết câu chúc |
| target / 2 | Đèn LED, “Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều.” | hoan_tien, cao, đèn bàn LED, tich_cuc | Đúng; 1.0 | Đúng; 1.0 | Hoà |

Hai ca thua lấy từ **regression**, được đánh dấu rõ, không gọi là ticket target. Ở các ca này model fine-tuned trả JSON phân loại thay vì thực hiện yêu cầu ngoài miền. Đây là biểu hiện cụ thể giải thích regression giảm. Baseline ở ca km cũng có diễn giải ký hiệu kilo chưa chính xác, nhưng trả số 1000 nên được keyword-recall chấm 1.0; thang đo đơn giản không đánh giá toàn bộ tính đúng của câu trả lời. Kết quả cho thấy cần cả benchmark và đọc thủ công, không chỉ dựa vào một điểm tổng hợp.

## 7. Kết luận và điều học được

Tôi chưa nên triển khai adapter như trợ lý đa dụng, vì nó không đáp ứng cổng regression dù điểm ticket cao. Fine-tuning chuyển được hành vi phân loại vào trọng số, cải thiện từ 0.765 lên 0.970 theo trường. Lợi ích này phải đặt cạnh giảm năng lực chung, tăng độ trễ và chi phí bảo trì adapter. Attention-only hoà text-linear cho thấy không thể chỉ dựa trên hướng dẫn lý thuyết để dự đoán thứ hạng. Cần khớp tham số, số bước và cách đánh giá rồi đo trực tiếp. Learning rate là đòn bẩy rõ trong ngân sách ngắn: thay một con số làm run không đạt format JSON. Mask đúng là điều kiện cần để thí nghiệm có ý nghĩa; dữ liệu một miền có thể làm model chuyên môn hoá quá mạnh. Token_stats gợi ý 256 nhắc tôi đo chiều dài trước khi tối ưu bộ nhớ. Với khách hàng thật, tôi sẽ thống nhất trước ngưỡng task, lỗi ngoài miền, latency và VRAM. Chỉ triển khai khi đạt cả các ngưỡng trên dữ liệu độc lập, không chọn riêng ví dụ đẹp để minh hoạ thành công. Chưa thể tuyên bố bản fine-tune tốt hơn về mọi mặt, hoặc biến verdict FAILED thành PASSED bằng việc bỏ kiểm tra regression.

Ba điều cụ thể: prompt B đạt format 100% mà chưa train; loss thấp hơn không giúp attn_only thắng target; giảm 56% VRAM bằng QLoRA đi kèm target thấp hơn và latency cao hơn.

Nếu có thêm hai giờ: thử replay data độc lập, giữ evaluation; thử max_length 256 và đo tốc độ/VRAM. Các phép thử này chưa thực hiện, không tính là kết quả.

## 8. Minh chứng

CP1–CP5 chạy đủ trên T4, 2815 giây (46.9 phút); Colab 119 unit test pass. `results/` lưu artifact gốc. NB6 và bonus chưa thực hiện. Adapter chuẩn đã tạo trong Colab; binary không đưa vào Git thông thường, xem ARTIFACTS.md.
