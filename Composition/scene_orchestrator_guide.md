## Yêu Cầu Cài Đặt

pip install numpy numpy-quaternion

## Cấu Trúc Dữ Liệu Đầu Vào

### 1. File JSON (ví dụ: `pos.json`)

```json
{
  "scene_001": [
    {
      "label": "Chair",
      "bounding_box": [0.5, 0.8, 0.5],
      "rotation": [1.0, 0.0, 0.0, 0.0],
      "center": [0.0, 0.4, 0.0]
    },
    {
      "label": "Table",
      "bounding_box": [1.2, 0.7, 1.2],
      "rotation": [1.0, 0.0, 0.0, 0.0],
      "center": [0.0, 0.35, 2.0]
    }
  ],
  "scene_002": [...]
}
```

**Giải thích các trường:**
- `label`: Tên object (phải trùng với tên file `.obj`)
- `bounding_box`: Kích thước mong muốn `[width, height, depth]`
- `rotation`: Quaternion `[w, x, y, z]`
- `center`: Vị trí trung tâm `[x, y, z]`

### 2. Thư Mục Objects

```
test/
├── Chair.obj
├── Chair.mtl
├── Table.obj
├── Table.mtl
├── Lighting.obj
└── Lighting.mtl
```

### Xuất Cảnh

```python
# Xuất một cảnh cụ thể
orchestrator.export_scene(
    scene_id="scene_001",
    output_path="output/scene_001.obj",
    include_floor=True  # Có tạo sàn hay không
)

# Xuất tất cả các cảnh
orchestrator.export_all_scenes(
    output_folder="output_scenes",
    include_floor=True
)
```