# M12 - Prewarm VieNeu cho câu cảnh báo

Nguồn: `output/demo_20261114/M12/prewarm/prewarm.json` (script `scripts/prewarm_forward_speech_20261114.py`). Provider `vieneu-onnx`, giọng `vieneu-HaiDang`. Thời gian tổng 7.9 s. Kết quả: 14/14 câu có audio, độ dài > 0: **đạt**.

Model được nạp từ `models/tts_hf_home` (cùng `HF_HOME` như `start.ps1`). Không có kết nối mạng.

| # | Câu | Độ dài (s) | Kích thước (byte) | File cache |
|---|---|---:|---:|---|
| 1 | Cẩn thận, người đi bộ phía trước | 1.8 | 21932 | `data/processed/tts_output/0be9bcec5e4839aeba21201af42959b2eee98a144da4094f1ac6f44b3c78ccfe.mp3` |
| 2 | Cẩn thận, xe đạp phía trước | 1.7 | 20780 | `data/processed/tts_output/98ffc2b3d02f78c6a2e58d2c6ebfb27feacec8940986cf87d42f0e9bbcda23df.mp3` |
| 3 | Cẩn thận, ô tô phía trước | 1.8 | 21932 | `data/processed/tts_output/570df48f0604417d152902bb0ba87830ae94a721e562662bfcc1708b37555016.mp3` |
| 4 | Cẩn thận, xe máy phía trước | 1.63 | 19916 | `data/processed/tts_output/e68e148e6f04123cd574ef4d9d2f94ebc188f9d3657c3f530567321f597235d3.mp3` |
| 5 | Cẩn thận, xe buýt phía trước | 1.87 | 22796 | `data/processed/tts_output/dea51a6348b6c4bdb9f7a1a4d238ac20dbd244d27c5d8e35f19a0622f2f7f002.mp3` |
| 6 | Cẩn thận, xe tải phía trước | 1.56 | 19052 | `data/processed/tts_output/aa9f5dbb12b51d22cb2067b2081eab3bca4d08f5723e80633d07c5e72bfb82f7.mp3` |
| 7 | Cẩn thận, phương tiện phía trước | 1.39 | 17036 | `data/processed/tts_output/203e593af050f25e3563eda775cbd9a913fedb264ad70cb6204f834ae6fbb0bb.mp3` |
| 8 | Chú ý người đi bộ | 1.08 | 13292 | `data/processed/tts_output/b386e9446fb1cd7c139e7774366dee5e7befbe5c38c6a29a8d1408a9226b57d8.mp3` |
| 9 | Chú ý xe đạp | 0.98 | 12140 | `data/processed/tts_output/6bb0804f1b7736f30745546601135cd7cad25c99c515fbc6c7653d1f3a7a087c.mp3` |
| 10 | Chú ý ô tô | 0.91 | 11276 | `data/processed/tts_output/ff581edc74761a86c391be0bcb66f6ee496fa3f85e01d7916eaed0983098852b.mp3` |
| 11 | Chú ý xe máy | 0.91 | 11276 | `data/processed/tts_output/897274ac4626fae33a48d0cd7ef0c5a4781ea9c11949f5870f3f5c1509165951.mp3` |
| 12 | Chú ý xe buýt | 0.91 | 11276 | `data/processed/tts_output/ae9619df5396b701274952bbd5b9f89938f2d4c647996663a0d7082ba0c78faa.mp3` |
| 13 | Chú ý xe tải | 0.98 | 12140 | `data/processed/tts_output/11b7e4f356c045273d26d80c563a533e87f02fa1b40402dcd28cb2f5d0dc79c9.mp3` |
| 14 | Chú ý phương tiện | 0.91 | 11276 | `data/processed/tts_output/56bce3c48e46999e0ec93e33849a81b8885af2eaeccfe42be13f63aa0c4363f9.mp3` |

Nhãn `Cẩn thận` (nguy_co) từ 1.39 đến 1.87 s, dưới 2 s. Nhãn `Chú ý` (chu_y) từ 0.91 đến 1.08 s.

Lưu ý: 6 câu `Chú ý` đã có sẵn trong cache trước đó, nên thời gian gần 0 s. Câu nguy_co được tổng hợp mới.
