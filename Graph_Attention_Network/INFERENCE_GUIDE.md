# Hướng Dẫn Sử Dụng Model GAT Layout Prediction

## 📋 Mô Tả

Model **LayoutGAT** là một mô hình Graph Attention Network (GAT) được thiết kế để dự đoán vị trí (position) và hướng (rotation) của các đối tượng trong một cảnh 3D dựa trên:
- **Thông tin đối tượng**: Loại đối tượng (class), kích thước bounding box
- **Quan hệ không gian**: Các mối quan hệ giữa các đối tượng (left_of, on_top_of, behind, etc.)

Model sử dụng kiến trúc GATv2 với nhiều layer attention để học các mối quan hệ phức tạp và sinh layout hợp lý cho scene.

---

## 🚀 Cài Đặt Môi Trường

### 1. Yêu Cầu Hệ Thống

- **Python**: 3.8 hoặc cao hơn
- **CUDA** (tùy chọn): Để tăng tốc độ inference với GPU
  - Kiểm tra CUDA: `nvidia-smi`
- **Hệ điều hành**: Windows, Linux, hoặc macOS

### 2. Cài Đặt Dependencies

```bash
# Cài đặt các package cần thiết
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install pyyaml tqdm numpy
```

**Lưu ý**: 
- Nếu dùng CPU, bỏ qua `--index-url` và cài torch thông thường
- Điều chỉnh phiên bản CUDA (`cu118`) phù hợp với máy của bạn

### 3. Kiểm Tra Cài Đặt

```python
import torch
import torch_geometric
print(f"PyTorch version: {torch.__version__}")
print(f"PyG version: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 📁 Cấu Trúc Thư Mục

```
gat_training_syn/
├── inference.py              # Script inference chính
├── configs/
│   └── configs_syn.yaml      # File cấu hình model
├── checkpoints/              # Thư mục chứa model đã train
│   ├── model_phase1234.pt    # Model full (khuyên dùng)
│   ├── model_phase123.pt
│   └── ...
├── src/
│   ├── model.py              # Định nghĩa LayoutGAT
│   ├── mappings.py           # Mapping class & relations
│   └── utils.py              # Utilities
├── examples/                 # (Tạo thư mục này để chứa ví dụ)
│   └── sample_scene.json     # File scene mẫu
└── INFERENCE_GUIDE.md        # File này
```

---

## 📝 Định Dạng Input: scene.json

### Cấu Trúc JSON

File `scene.json` cần có cấu trúc như sau:

```json
{
  "objects": [
    {
      "id": 0,
      "label": "Bed",
      "normalized_bounding_box": [0.9876, 0.3688, 0.7973],
      "normalized_relative_center": [0.0, 0.0, 0.0],
      "rot": [1.0, 0.0, 0.0, 0.0]
    },
    {
      "id": 1,
      "label": "Chair",
      "normalized_bounding_box": [0.4435, 0.9959, 0.3451]
    }
  ],
  "relationships": [
    {
      "obj_id1": 1,
      "obj_id2": 0,
      "relation": "left_of"
    }
  ]
}
```

### Mô Tả Các Field

#### **objects** (bắt buộc)
Danh sách các đối tượng trong scene.

- **`id`** (int, bắt buộc): ID duy nhất của đối tượng (bắt đầu từ 0)
- **`label`** (string, bắt buộc): Loại đối tượng. Các giá trị hợp lệ:
  ```
  "Armchair", "Banana", "Basket", "Bed", "Cabinet", "Cake", "Carpet", 
  "Chair", "Cup", "Easel", "Fan", "Fridge", "Floor_lamp", "Laptop_close", 
  "Laptop_open", "Lighting", "Mirror", "Monkey", "Panda", "Piano", 
  "Picture", "Pier", "Pillow", "Rabbit", "Sofa", "TV", "Table", "Test", 
  "Toy", "Tree", "Vase", "Wardrobe", "Washingmachine", "human_lying", 
  "human_sitting", "human_standing", "Other"
  ```
  *(Nếu label không có trong danh sách, sẽ được map sang "Other")*

- **`normalized_bounding_box`** (array[3], bắt buộc): Kích thước bounding box đã chuẩn hóa
  - Format: `[width, height, depth]`
  - Giá trị từ 0.0 đến ~1.0 (tỷ lệ so với kích thước phòng)
  - Ví dụ: `[0.5, 0.3, 0.4]` = 50% chiều rộng phòng, 30% chiều cao, 40% chiều sâu

- **`normalized_relative_center`** (array[3], optional): Vị trí ban đầu (nếu có)
  - Format: `[x, y, z]`
  - Thường dùng cho anchor object (object đầu tiên, id=0)
  - **Anchor nên đặt tại `[0.0, 0.0, 0.0]`** để model hoạt động tốt nhất

- **`rot`** (array[4], optional): Rotation ban đầu dưới dạng quaternion
  - Format: `[w, x, y, z]`
  - Ví dụ: `[1.0, 0.0, 0.0, 0.0]` = không rotation

- **`name`** (string, optional): Tên đối tượng (để dễ đọc kết quả)

#### **relationships** (optional, nhưng nên có)
Danh sách các mối quan hệ không gian giữa các đối tượng.

- **`obj_id1`** (int, bắt buộc): ID của đối tượng thứ nhất
- **`obj_id2`** (int, bắt buộc): ID của đối tượng thứ hai  
- **`relation`** (string, bắt buộc): Loại quan hệ. Các giá trị hợp lệ:
  ```
  "on_top_of", "above", "under", "left_of", "right_of", 
  "in_front_of", "behind", "facing", "in"
  ```

**Lưu ý**:
- Model tự động tạo bidirectional edges (cạnh hai chiều)
- Nếu không có relationships, model vẫn chạy nhưng kết quả kém chính xác

---

## 📤 Định Dạng Output

### Cấu Trúc JSON Output

File output (mặc định: `<scene>_predictions.json`) có cấu trúc:

```json
{
  "metadata": {
    "model": "LayoutGAT",
    "num_objects": 2,
    "device": "cuda"
  },
  "predictions": [
    {
      "id": 0,
      "name": "Bed_0",
      "label": "Bed",
      "input": {
        "bounding_box": [0.9876, 0.3688, 0.7973],
        "position": [0.0, 0.0, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0]
      },
      "prediction": {
        "position": [0.0, 0.0, 0.0],
        "rotation_quaternion": [0.9998, -0.0045, 0.0189, 0.0021],
        "rotation_format": "xyzw"
      }
    },
    {
      "id": 1,
      "name": "Chair_1",
      "label": "Chair",
      "input": {
        "bounding_box": [0.4435, 0.9959, 0.3451]
      },
      "prediction": {
        "position": [-1.2534, 0.0123, 0.4567],
        "rotation_quaternion": [0.7071, 0.0, 0.7071, 0.0],
        "rotation_format": "xyzw"
      }
    }
  ]
}
```

### Mô Tả Output

- **`position`**: Vị trí dự đoán trong không gian 3D
  - Format: `[x, y, z]`
  - Giá trị trong khoảng [-2.0, 2.0] (tùy room_scale trong config)
  - Đơn vị: tương đối so với kích thước phòng chuẩn hóa

- **`rotation_quaternion`**: Rotation dự đoán dưới dạng quaternion
  - Format: `[w, x, y, z]`
  - Đã được normalize (|q| = 1)

---

## 🎯 Cách Sử Dụng

### Phương Pháp 1: Command Line (Khuyên Dùng)

```bash
# Cú pháp cơ bản
python inference.py --scene <path_to_scene.json> --checkpoint <path_to_checkpoint.pt>

# Ví dụ đầy đủ
python inference.py \
    --scene examples/sample_scene.json \
    --checkpoint checkpoints/model_phase1234.pt \
    --config configs/configs_syn.yaml \
    --output results/my_predictions.json
```

**Các tham số**:
- `--scene` (bắt buộc): Đường dẫn đến file scene.json
- `--checkpoint`: Đường dẫn checkpoint (mặc định: `checkpoints/model_phase1234.pt`)
- `--config`: Đường dẫn config (mặc định: `configs/configs_syn.yaml`)
- `--output`: Đường dẫn file output (mặc định: `<scene>_predictions.json`)

### Phương Pháp 2: Python Script

```python
from inference import SceneInference

# Khởi tạo inferencer
inferencer = SceneInference(
    checkpoint_path='checkpoints/model_phase1234.pt',
    config_path='configs/configs_syn.yaml'
)

# Run inference
predictions = inferencer.run_inference(
    scene_json_path='examples/sample_scene.json',
    output_path='results/output.json'
)

# Predictions trả về tuple (positions, rotations)
positions, rotations = predictions
print(f"Predicted {len(positions)} objects")
```

### Phương Pháp 3: Batch Processing

```python
import glob
from inference import SceneInference

inferencer = SceneInference('checkpoints/model_phase1234.pt')

# Process tất cả scene trong thư mục
scene_files = glob.glob('examples/*.json')
for scene_file in scene_files:
    print(f"\nProcessing: {scene_file}")
    inferencer.run_inference(scene_file)
```

---

## 📊 Ví Dụ Thực Tế

### Ví Dụ 1: Scene Đơn Giản (2 Objects)

**Input**: `examples/simple_scene.json`
```json
{
  "objects": [
    {
      "id": 0,
      "label": "Table",
      "normalized_bounding_box": [0.8, 0.4, 0.6],
      "normalized_relative_center": [0.0, 0.0, 0.0],
      "rot": [1.0, 0.0, 0.0, 0.0]
    },
    {
      "id": 1,
      "label": "Chair",
      "normalized_bounding_box": [0.45, 0.95, 0.35]
    }
  ],
  "relationships": [
    {
      "obj_id1": 1,
      "obj_id2": 0,
      "relation": "in_front_of"
    }
  ]
}
```

**Chạy**:
```bash
python inference.py --scene examples/simple_scene.json
```

**Output Console**:
```
Loading model from checkpoints/model_phase1234.pt...
Model loaded successfully on cuda
Loaded scene with 2 objects and 1 relationships
Converting scene to graph...
Graph: 2 nodes, 2 edges
Running inference...

============================================================
PREDICTION RESULTS
============================================================

Object 0: Table
  Position: [0.0000, 0.0000, 0.0000]
  Rotation: [0.9987, 0.0023, -0.0145, 0.0489]

Object 1: Chair
  Position: [-0.8523, 0.0234, -1.2341]
  Rotation: [0.7123, 0.0034, 0.7018, -0.0012]
============================================================

Results saved to: examples/simple_scene_predictions.json
```

### Ví Dụ 2: Scene Phức Tạp (Living Room)

**Input**: `examples/living_room.json`
```json
{
  "objects": [
    {"id": 0, "label": "Sofa", "normalized_bounding_box": [1.2, 0.5, 0.8]},
    {"id": 1, "label": "TV", "normalized_bounding_box": [0.7, 0.4, 0.05]},
    {"id": 2, "label": "Table", "normalized_bounding_box": [0.6, 0.3, 0.6]},
    {"id": 3, "label": "Carpet", "normalized_bounding_box": [1.5, 0.01, 1.2]}
  ],
  "relationships": [
    {"obj_id1": 1, "obj_id2": 0, "relation": "in_front_of"},
    {"obj_id1": 2, "obj_id2": 0, "relation": "in_front_of"},
    {"obj_id1": 3, "obj_id2": 0, "relation": "under"},
    {"obj_id1": 3, "obj_id2": 2, "relation": "under"}
  ]
}
```

---

## ⚙️ Cấu Hình Model (configs_syn.yaml)

Các tham số quan trọng:

```yaml
model:
  num_classes: 37           # Số lượng class objects
  embedding_dim: 64         # Dimension của class embedding
  hidden_dim: 256           # Hidden dimension của GAT
  num_heads: 4              # Số attention heads
  num_gat_layers: 4         # Số layer GAT
  room_scale: 2.0           # Scale của output position ([-2, 2])
  dropout_rate: 0.1

training:
  device: "cuda"            # "cuda" hoặc "cpu"
```

**Lưu ý**: Không nên thay đổi các tham số này trừ khi bạn muốn retrain model.

---

## 🐛 Troubleshooting

### Lỗi: "CUDA out of memory"
**Giải pháp**:
```bash
# Sử dụng CPU thay vì GPU
# Sửa trong configs/configs_syn.yaml:
training:
  device: "cpu"
```

### Lỗi: "Unknown class label"
**Giải pháp**: Kiểm tra lại `label` trong scene.json. Nếu không có trong danh sách CLASS_MAPPING, sẽ tự động map sang "Other".

### Lỗi: "No module named 'torch_geometric'"
**Giải pháp**:
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Kết quả prediction không tốt
**Nguyên nhân & Giải pháp**:
1. **Thiếu relationships**: Thêm nhiều quan hệ không gian giữa objects
2. **Anchor không đúng**: Đảm bảo object đầu tiên (id=0) có position [0,0,0]
3. **Bounding box không chuẩn**: Kiểm tra lại normalization (giá trị 0-1)
4. **Checkpoint không phù hợp**: Thử các checkpoint khác (phase1, phase123, phase1234)

---

## 📈 Model Checkpoints

Các checkpoint có sẵn:

| Checkpoint | Mô Tả | Khuyên Dùng |
|-----------|-------|-------------|
| `model_phase1.pt` | Base model, train trên Phase 1 | Scene đơn giản (2-3 objects) |
| `model_phase123.pt` | Train lũy tiến Phase 1→2→3 | Scene trung bình (3-5 objects) |
| `model_phase1234.pt` | Full model, train trên tất cả Phase | **Scene phức tạp (5+ objects)** ⭐ |
| `model_phase123_finetune.pt` | Fine-tuned từ Phase 123 | Alternate choice |

**Khuyến nghị**: Sử dụng `model_phase1234.pt` cho kết quả tốt nhất.

---

## 🔧 Nâng Cao

### Tùy Chỉnh Class Mapping

Nếu muốn thêm class mới, chỉnh sửa `src/mappings.py`:

```python
CLASS_MAPPING = {
    # ... existing classes ...
    "MyNewClass": 37,  # ID tiếp theo
    "Other": 38        # Update Other ID
}
NUM_CLASSES = 39  # Update total
```

**Lưu ý**: Sau khi thay đổi cần **retrain model**.

### Visualization (Tùy Chọn)

Để visualize kết quả prediction trong 3D:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load predictions
with open('output_predictions.json', 'r') as f:
    data = json.load(f)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for obj in data['predictions']:
    pos = obj['prediction']['position']
    ax.scatter(pos[0], pos[1], pos[2], s=100, label=obj['label'])
    ax.text(pos[0], pos[1], pos[2], obj['name'])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
```

---

## 📚 References & Citation

Model này sử dụng:
- **GATv2** (Graph Attention Networks v2): Brody et al., 2021
- **PyTorch Geometric**: Fey & Lenssen, 2019

Nếu sử dụng model này trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@misc{layoutgat2025,
  title={LayoutGAT: Graph Attention Networks for 3D Scene Layout Generation},
  author={Your Name},
  year={2025}
}
```

---

## 📞 Liên Hệ & Hỗ Trợ

Nếu gặp vấn đề hoặc có câu hỏi:
1. Kiểm tra lại **Troubleshooting** section
2. Xem lại **Định Dạng Input** để đảm bảo JSON đúng format
3. Thử với **ví dụ đơn giản** trước khi test scene phức tạp

---

## 📄 License

Project này chỉ dùng cho mục đích học tập và nghiên cứu.

---

**Happy Inferencing! 🎉**
