# Quick Start - Model Inference

## 🚀 Chạy Nhanh

### 1. Cài đặt (chỉ cần 1 lần)
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pyyaml tqdm numpy
```

### 2. Chạy inference với file mẫu
```bash
# Scene đơn giản (3 objects)
python inference.py --scene examples/sample_scene.json

# Living room (5 objects)
python inference.py --scene examples/living_room.json

# Với checkpoint và output tùy chỉnh
python inference.py \
    --scene examples/living_room.json \
    --checkpoint checkpoints/model_phase1234.pt \
    --output results/my_predictions.json
```

### 3. Kiểm tra kết quả
Kết quả sẽ được lưu tại: `examples/sample_scene_predictions.json`

---

## 📝 Tạo Scene Của Bạn

Tạo file `my_scene.json`:

```json
{
  "objects": [
    {
      "id": 0,
      "label": "Bed",
      "normalized_bounding_box": [1.0, 0.4, 0.8]
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
      "relation": "left_of"
    }
  ]
}
```

Chạy:
```bash
python inference.py --scene my_scene.json
```

---

## 📚 Chi Tiết Đầy Đủ

Xem file **[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)** để biết:
- Định dạng input/output chi tiết
- Danh sách đầy đủ các class và relation
- Troubleshooting
- Ví dụ nâng cao

---

## 🎯 Các Object Class Hỗ Trợ

```
Armchair, Banana, Basket, Bed, Cabinet, Cake, Carpet, Chair, Cup, 
Easel, Fan, Fridge, Floor_lamp, Laptop_close, Laptop_open, Lighting, 
Mirror, Monkey, Panda, Piano, Picture, Pier, Pillow, Rabbit, Sofa, 
TV, Table, Test, Toy, Tree, Vase, Wardrobe, Washingmachine, 
human_lying, human_sitting, human_standing, Other
```

## 🔗 Các Relation Hỗ Trợ

```
on_top_of, above, under, left_of, right_of, 
in_front_of, behind, facing, in
```

---

## ⚡ Tips

1. **Object đầu tiên (id=0)** nên đặt tại `[0, 0, 0]` - là anchor
2. **Bounding box** đã normalize (giá trị 0-1, tỷ lệ với phòng)
3. **Thêm nhiều relationships** để model predict chính xác hơn
4. Dùng **model_phase1234.pt** cho kết quả tốt nhất

---

**Cần trợ giúp?** → Xem [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
