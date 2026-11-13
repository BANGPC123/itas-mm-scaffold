# M4 - Audit nhánh và evidence metadata

So với `main` tại `df47efab3345693a45284e66bef83709fbc2a554`. Checkpoint audit `84b215792dac2ee26ef9a080afa2b72267a75ba3`. Tác giả và ngày trong bảng là của tip. Ahead/behind lấy từ `git rev-list --left-right --count main...<ref>`. Tất cả tip origin đã đối chiếu với `git ls-remote --heads origin`. `origin/HEAD` là alias.

Nguồn số liệu cục bộ trong `output/demo_20261114/M4/` của repo chính gồm `branches.tsv`, `origin-heads.tsv`, `merge-tree.log`, `file-classification.tsv`, `pytest-cpu.log`, `legal-before.log` và `legal-index.json`.

## Nhánh

| Nhánh | Ahead | Behind | Trạng thái | Tip | Tác giả | Ngày |
| --- | --- | --- | --- | --- | --- | --- |
| backend/system-refactor | 8 | 97 | Chưa nhập | 1e9d7f9e | Onsra | 2026-09-27 |
| backup/kiet-final-evidence | 0 | 55 | Đã nhập | 15cdef5a | Tran Anh Kiet | 2026-10-29 |
| chore/itas-mm-repo-hygiene-20261014 | 0 | 98 | Đã nhập | 2b3f36da | Khánh Băng | 2026-10-14 |
| experiment/lane-preprocessing-20261102 | 0 | 8 | Đã nhập | 5d84fd28 | Tran Anh kiet | 2026-11-03 |
| feature/itas-mm-concordance-lane-fusion-20261014 | 0 | 93 | Đã nhập | a1337ab0 | Khánh Băng | 2026-10-14 |
| feature/itas-mm-import-comment-yolo11-20261014 | 0 | 95 | Đã nhập | 56be1196 | Khánh Băng | 2026-10-14 |
| feature/itas-mm-offline-video-accuracy-20261014 | 0 | 92 | Đã nhập | 6d617864 | Khánh Băng | 2026-10-14 |
| feature/itas-mm-precision-offline-vn-sign-lane-20261014 | 0 | 89 | Đã nhập | c2b3e2ab | Khánh Băng | 2026-10-14 |
| feature/itas-mm-validated-video-car-20261014 | 0 | 97 | Đã nhập | f7014c4f | Khánh Băng | 2026-10-14 |
| fix/lane-simplified-demo-20261102 | 0 | 11 | Đã nhập | ed0cbc74 | Tran Anh kiet | 2026-11-10 |
| main | 0 | 0 | main | df47efab | Tran Anh kiet | 2026-11-10 |
| research/itas-mm-ai-visual-review-20261014 | 0 | 96 | Đã nhập | 71571aa5 | Khánh Băng | 2026-10-14 |
| wip/task6-legal-retrieval-benchmark-20261001 | 39 | 97 | Chưa nhập | 1ac9619e | Onsra | 2026-10-02 |
| work/audit-20261114 | 1 | 0 | Chưa nhập | 84b21579 | Tran Anh kiet | 2026-11-13 |
| work/bench-20261112 | 1 | 0 | Chưa nhập | 84b21579 | Tran Anh kiet | 2026-11-13 |
| work/demo-20261114 | 1 | 0 | Chưa nhập | 84b21579 | Tran Anh kiet | 2026-11-13 |
| work/final-20261102 | 0 | 20 | Đã nhập | 358c3c37 | Tran Anh kiet | 2026-11-02 |
| work/frame-review-20261015 | 0 | 92 | Đã nhập | 6d617864 | Khánh Băng | 2026-10-14 |
| work/kiet-dataset-v2 | 0 | 50 | Đã nhập | d2335bb6 | Tran Anh kiet | 2026-10-29 |
| work/kiet-final-evidence | 0 | 53 | Đã nhập | 5db89382 | Tran Anh kiet | 2026-10-29 |
| work/kiet-pipeline-fixes | 0 | 72 | Đã nhập | dc51f07e | Tran Anh kiet | 2026-10-28 |
| work/kiet-recognizer-v2 | 0 | 57 | Đã nhập | 5190f97e | Tran Anh kiet | 2026-10-29 |
| origin/backend/system-refactor | 8 | 97 | Chưa nhập | 1e9d7f9e | Onsra | 2026-09-27 |
| origin/chore/itas-mm-repo-hygiene-20261014 | 0 | 98 | Đã nhập | 2b3f36da | Khánh Băng | 2026-10-14 |
| origin/feature/itas-mm-concordance-lane-fusion-20261014 | 0 | 93 | Đã nhập | a1337ab0 | Khánh Băng | 2026-10-14 |
| origin/feature/itas-mm-import-comment-yolo11-20261014 | 0 | 95 | Đã nhập | 56be1196 | Khánh Băng | 2026-10-14 |
| origin/feature/itas-mm-offline-video-accuracy-20261014 | 0 | 92 | Đã nhập | 6d617864 | Khánh Băng | 2026-10-14 |
| origin/feature/itas-mm-precision-offline-vn-sign-lane-20261014 | 0 | 89 | Đã nhập | c2b3e2ab | Khánh Băng | 2026-10-14 |
| origin/feature/itas-mm-validated-video-car-20261014 | 0 | 97 | Đã nhập | f7014c4f | Khánh Băng | 2026-10-14 |
| origin/feature/itas-mm-weights-lfs-20261018 | 0 | 88 | Đã nhập | 478ad7a8 | Khánh Băng | 2026-10-18 |
| origin/integration/itas-mm-release-candidate-20261021 | 0 | 85 | Đã nhập | a5cf0416 | Onsra | 2026-10-22 |
| origin/main | 0 | 2 | Đã nhập | b7e341e1 | Tran Anh kiet | 2026-11-10 |
| origin/research/itas-mm-ai-visual-review-20261014 | 0 | 96 | Đã nhập | 71571aa5 | Khánh Băng | 2026-10-14 |
| origin/wip/task6-legal-retrieval-benchmark-20261001 | 39 | 97 | Chưa nhập | 1ac9619e | Onsra | 2026-10-02 |

Các hàng `Đã nhập` là ứng viên dọn theo ancestry so với main. Giữ ref mặc định, nhánh đang dùng và nhánh backup nếu còn cần rollback. Tip backend là tổ tiên của tip WIP. Các nhánh benchmark, audit và demo cũng còn commit riêng tại thời điểm chụp bảng.

## Phép mô phỏng hợp nhất

Dùng `git merge-tree --write-tree --messages main origin/wip/task6-legal-retrieval-benchmark-20261001` từ worktree audit, với object directory riêng dưới M4. Cách này giữ giới hạn một worktree và vẫn chạy phép hợp nhất ba phía. Exit 1. Không có trạng thái merge cần abort.

- CONFLICT (add/add): Merge conflict in .gitattributes
- CONFLICT (content): Merge conflict in README.md
- CONFLICT (content): Merge conflict in configs/interaction.yaml
- CONFLICT (content): Merge conflict in configs/perception.yaml
- CONFLICT (add/add): Merge conflict in data/regulations/corpus.json
- CONFLICT (content): Merge conflict in src/reasoning/document_loader.py
- CONFLICT (content): Merge conflict in tests/integration/test_pipeline_e2e.py

## Phân loại file

Danh sách là diff từ merge-base đến tip WIP. Hai tip được đối chiếu theo blob và mode cho từng file.

| File | Phân loại | Lý do |
| --- | --- | --- |
| `.env.example` | KHÔNG PORT | Nhánh xóa file mẫu môi trường. |
| `.gitattributes` | KHÔNG PORT | Giữ thuộc tính LFS của main. |
| `README.md` | MAIN MỚI HƠN | Giữ hướng dẫn demo và vận hành hiện tại. |
| `THIRD_PARTY_NOTICES.md` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `config/regulations.json` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `configs/interaction.yaml` | MAIN MỚI HƠN | Giữ cấu hình tương tác và TTS hiện tại. |
| `configs/perception.yaml` | MAIN MỚI HƠN | Giữ pipeline, trọng số và cấu hình demo hiện tại. |
| `configs/reasoning.yaml` | MAIN MỚI HƠN | Giữ cấu hình reasoning và validated_corpus: false. |
| `data/evaluation/legal_retrieval.json` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `data/regulations/corpus.json` | KHÔNG PORT | Corpus khác phiên bản và ngoài phạm vi port. |
| `data/regulations/raw/decree-168-2024-nd-cp/source.html` | KHÔNG PORT | Snapshot nguồn corpus, dành cho audit provenance riêng. |
| `data/regulations/raw/law-36-2024-qh15/source.json` | KHÔNG PORT | Snapshot nguồn corpus, dành cho audit provenance riêng. |
| `data/regulations/raw/qcvn-41-2024-bgtvt/source.pdf` | KHÔNG PORT | Snapshot nguồn corpus, dành cho audit provenance riêng. |
| `data/regulations/sample_traffic_rules.md` | KHÔNG PORT | Nhánh xóa mẫu cũ. Giữ file của main. |
| `docs/superpowers/plans/2026-09-27-backend-system-refactor.md` | KHÔNG PORT | Spec hoặc plan tháng 9, ngoài phạm vi port. |
| `docs/superpowers/plans/2026-09-29-legal-corpus-v1.md` | KHÔNG PORT | Spec hoặc plan tháng 9, ngoài phạm vi port. |
| `docs/superpowers/specs/2026-09-26-backend-system-refactor-design.md` | KHÔNG PORT | Spec hoặc plan tháng 9, ngoài phạm vi port. |
| `docs/superpowers/specs/2026-09-28-legal-corpus-v1-design.md` | KHÔNG PORT | Spec hoặc plan tháng 9, ngoài phạm vi port. |
| `requirements.txt` | MAIN MỚI HƠN | Giữ dependency hiện tại, bao gồm Pillow cho API. |
| `scripts/build_vector_index.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `scripts/evaluate_legal_retrieval.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `scripts/fetch_regulations.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `src/reasoning/document_loader.py` | MAIN MỚI HƠN | Giữ hỗ trợ nguồn và loader của main. |
| `src/reasoning/legal_normalizer.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `src/reasoning/ollama_client.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `src/reasoning/rag_chain.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `src/reasoning/vector_store.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `src/utils/types.py` | CHỈ CÓ Ở NHÁNH - NÊN PORT | Chỉ hunk kiểu DocumentChunk còn thiếu. Nếu mở lại, phải dùng list[str] | list[DocumentChunk] để giữ cả luồng offline hiện tại. Gate M4.3 đang chặn. |
| `tests/integration/test_pipeline_e2e.py` | MAIN MỚI HƠN | Giữ kiểm thử theo pipeline hiện tại. |
| `tests/unit/test_build_vector_index.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `tests/unit/test_corpus_fingerprint.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `tests/unit/test_document_loader.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `tests/unit/test_fetch_regulations.py` | KHÔNG PORT | Chỉ có ở nhánh, gắn với quy trình lấy corpus cũ. |
| `tests/unit/test_legal_normalizer.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `tests/unit/test_ollama_client.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `tests/unit/test_rag_chain.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `tests/unit/test_retrieval_evaluation.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `tests/unit/test_script_entrypoints.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |
| `tests/unit/test_vector_store.py` | ĐÃ CÓ TRÊN MAIN (giống hệt) | Cùng blob và mode. |

## Quyết định port

Không có file được port. `vector_store.py`, `rag_chain.py` và hai test đã có nội dung giống hệt tip WIP. Logic giữ metadata và kiểm tra alias đã có ở checkpoint. Hunk `ReasoningResult.retrieved_chunks` còn lệch, nhưng dùng nguyên kiểu của nhánh sẽ bỏ trường hợp `list[str]` của `OfflineLegalRetriever` và hợp đồng API hiện tại.

- Pytest CPU trên checkpoint: 617 passed, 6 failed, 11 skipped, 38 deselected. Hai lỗi thiếu manifest dataset, hai lỗi thiếu mã UFLD local và hai lỗi thiếu index luật offline. Lần đầu gặp `PermissionError: [WinError 5]` ở thư mục tạm pytest. Lần chạy lại trong thư mục riêng ghi kết quả đầy đủ ở `pytest-cpu.log`.
- Benchmark 24 ca: chưa đo. Script dừng với `chromadb.errors.NotFoundError: Collection [traffic_regulations] does not exist` ở thư mục chạy cô lập. Index ở repo chính có collection nhưng có 0 embedding theo truy vấn SQLite chỉ đọc. Không có phép so sánh trước/sau.

Đề xuất cho Tiến: giữ phần truy xuất và test đã có ở main. Nếu tiếp tục corpus WIP, đối chiếu provenance và khác biệt locator theo từng nguồn, phục hồi index benchmark có fingerprint và chạy lại bộ 24 ca trước khi đưa ra thay đổi riêng.
