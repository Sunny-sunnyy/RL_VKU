# Session Handoff: Pong RL Benchmarking cleanest context reloading

Copy và paste toàn bộ nội dung dưới đây vào ô chat của phiên làm việc (New Chat session) mới để AI ngay lập tức tải và nắm bắt 100% ngữ cảnh sạch sẽ của dự án.

---

```markdown
Chào bạn, tôi là sinh viên đang thực hiện đề tài cuối kỳ so sánh thuật toán Reinforcement Learning trên Atari Pong. Dự án nằm hoàn toàn trong thư mục `G:\RL_VKU\ck\pong-comparison`.

Hiện tại dự án vừa hoàn tất việc tích hợp thuật toán PPO (Scratch) tự viết từ đầu bằng PyTorch vào bộ công cụ đánh giá so sánh và vẽ biểu đồ hiệu năng chung. Để thiết lập lại ngữ cảnh sạch sẽ và nắm vững dự án, bạn hãy thực hiện nghiêm ngặt các bước sau:

1. Đọc và nắm vững master handoff trong file [CLAUDE.md](file:///G:/RL_VKU/ck/pong-comparison/CLAUDE.md) để hiểu cấu trúc thư mục mới và các lệnh chạy virtual environment.
2. Đọc [walkthrough.md](file:///G:/RL_VKU/ck/pong-comparison/walkthrough.md) để hiểu chi tiết toán học của thuật toán PPO Scratch mới xây dựng (rollout, GAE, clipped loss), cách tích hợp nạp mô hình checkpoint PPO Scratch và kết quả unit tests.
3. Đọc [progress_log.md](file:///G:/RL_VKU/ck/pong-comparison/progress_log.md) để nắm được các milestones đã hoàn thành (gồm cả Phase 7 tích hợp PPO Scratch vào evaluation suite).
4. Chạy kiểm thử tự động bằng lệnh `uv run pytest` trong thư mục dự án để xác nhận toàn bộ 12/12 unit tests tiếp tục hoạt động ổn định và chính xác 100%.
5. Thử chạy kịch bản vẽ đồ thị đối chứng `uv run scripts/evaluate.py --plot-only` để kiểm tra khả năng tích hợp.
6. Sau khi đọc xong, hãy tóm tắt cực kỳ ngắn gọn những gì bạn hiểu về trạng thái dự án hiện tại và hỏi tôi về bước ưu tiên huấn luyện hoặc đánh giá tiếp theo bạn cần hỗ trợ.

Lưu ý:
- Giao tiếp bằng Tiếng Việt (giữ nguyên các thuật ngữ Tiếng Anh chuyên ngành).
- Code và comments bằng Tiếng Anh chuẩn.
- Tuyệt đối KHÔNG sử dụng emoji trong code, comments, print statements hay bất kỳ nhật ký, báo cáo nào.
- Tuân thủ quy tắc làm việc incremental, viết code hoàn chỉnh không dùng placeholders (no pass, no TODO).
```
