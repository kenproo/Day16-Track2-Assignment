# Báo cáo kết quả benchmark LightGBM trên AWS CPU

1. Mô hình LightGBM được huấn luyện trên 227.845 mẫu của bộ Credit Card Fraud Detection bằng máy AWS có 2 vCPU và khoảng 4 GB RAM.
2. Thời gian huấn luyện là 10,9106 giây, cho thấy mô hình có thể được huấn luyện nhanh trên cấu hình CPU `t3.medium` của bài lab.
3. AUC-ROC đạt 0,883917, thể hiện khả năng phân biệt giao dịch gian lận và bình thường ở mức tốt.
4. Accuracy đạt 99,8789%, nhưng do dữ liệu rất mất cân bằng nên F1-score 0,666667 là chỉ số phản ánh thực tế hơn.
5. Precision đạt 0,633028 và recall đạt 0,704082, nghĩa là mô hình phát hiện được khoảng 70,4% giao dịch gian lận trong tập kiểm thử.
6. Độ trễ suy luận trung bình cho một mẫu là 1,292315 ms, phù hợp với các tác vụ dự đoán gần thời gian thực trên CPU.
7. Với batch 1.000 mẫu, thời gian suy luận là 15,0529 ms và throughput đạt 66.432,30 mẫu/giây, cho thấy xử lý theo lô rất hiệu quả.
8. Ảnh giám sát sau benchmark cho thấy CPU gần như nhàn rỗi và còn khoảng 3,2 GB RAM khả dụng; Cost Explorer tại thời điểm chụp vẫn ghi nhận tổng chi phí 0,00 USD.
