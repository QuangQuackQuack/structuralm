import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import quaternion  # pip install numpy-quaternion

class SceneOrchestrator:
    def __init__(self, floor_size: Tuple[float, float] = (5.0, 5.0)):
        self.floor_size = floor_size
        self.scenes = {}
        self.scene_floor_heights = {}
        self.scene_materials = {}

        # Pre-rotation quaternion cố định
        #self.pre_rotation = np.quaternion(0.6899, -0.6899, 0.0, 0.0).normalized()
        self.pre_rotation = np.quaternion(0.6533, -0.6533, 0.2706, 0.2706).normalized()

    def load_obj_with_mtl(self, obj_path: str, object_label: str) -> Dict:
        """
        Load obj đầy đủ (v, vt, vn, faces, materials) và đọc file mtl đi kèm.
        Đổi tên material để tránh trùng lặp (VD: 'Mat1' -> 'Chair_Mat1').
        """
        path = Path(obj_path)
        mtl_path = path.with_suffix('.mtl')

        # 1. Đọc và xử lý file .mtl (nếu có)
        materials_content = []
        material_mapping = {} # Old Name -> New Name

        if mtl_path.exists():
            with open(mtl_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('newmtl '):
                        old_name = line.split()[1]
                        new_name = f"{object_label}_{old_name}" # Unique name
                        material_mapping[old_name] = new_name
                        materials_content.append(f"newmtl {new_name}")
                    else:
                        materials_content.append(line)
        else:
            print(f"Warning: No .mtl file found for {obj_path}")

        # 2. Đọc file .obj
        vertices = []
        normals = []
        uvs = []
        faces = [] # List of dict: {'mat': 'MaterialName', 'indices': [[v, vt, vn], ...]}

        current_mat = None

        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('vn '):
                    parts = line.split()
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('vt '):
                    parts = line.split()
                    uvs.append([float(parts[1]), float(parts[2])]) # UV thường chỉ có 2 coords
                elif line.startswith('usemtl '):
                    raw_mat = line.split()[1]
                    # Map sang tên mới unique
                    current_mat = material_mapping.get(raw_mat, f"{object_label}_{raw_mat}")
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    # Face format: v/vt/vn or v//vn or v/vt
                    face_indices = []
                    for p in parts:
                        vals = p.split('/')
                        v_idx = int(vals[0]) - 1
                        vt_idx = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else None
                        vn_idx = int(vals[2]) - 1 if len(vals) > 2 and vals[2] else None
                        face_indices.append((v_idx, vt_idx, vn_idx))

                    faces.append({
                        'material': current_mat,
                        'indices': face_indices
                    })

        return {
            'vertices': np.array(vertices, dtype=np.float32),
            'normals': np.array(normals, dtype=np.float32) if normals else None,
            'uvs': np.array(uvs, dtype=np.float32) if uvs else None,
            'faces': faces,
            'materials_content': materials_content
        }

    def transform_object(self, obj_data: Dict, bounding_box: List[float],
                         rotation_quat: np.ndarray, center: np.ndarray):
        """
        Transform Vertices và Normals với pre-rotation.
        Rotation pipeline: pre_rotation -> input_rotation
        """
        vertices = obj_data['vertices']
        normals = obj_data['normals']

        # --- 1. Vertices Transformation ---
        vertices_centered = vertices - np.mean(vertices, axis=0)

        # Scale
        min_coords = np.min(vertices_centered, axis=0)
        max_coords = np.max(vertices_centered, axis=0)
        current_bbox = max_coords - min_coords
        current_bbox = np.where(current_bbox == 0, 1.0, current_bbox)
        scale_factors = np.array(bounding_box) / current_bbox
        transformed_v = vertices_centered * scale_factors

        # Rotate: Áp dụng pre-rotation trước, sau đó input rotation
        q_input = np.quaternion(*rotation_quat).normalized()
        q_combined = q_input * self.pre_rotation  # Nhân quaternion: input sau * pre trước
        rot_mat = quaternion.as_rotation_matrix(q_combined)
        transformed_v = transformed_v @ rot_mat.T

        # Translate
        transformed_v = transformed_v + center

        obj_data['vertices'] = transformed_v

        # --- 2. Normals Transformation ---
        # Normals chỉ cần Rotate (không Scale theo cách thường, không Translate)
        if normals is not None and len(normals) > 0:
            # Rotate normals với cùng combined quaternion
            transformed_n = normals @ rot_mat.T
            # Normalize lại để đảm bảo độ dài = 1
            norms = np.linalg.norm(transformed_n, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms) # Tránh chia 0
            obj_data['normals'] = transformed_n / norms

    def _get_world_aabb(self, vertices: np.ndarray) -> Dict[str, float]:
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        return {
            'min_x': min_coords[0], 'max_x': max_coords[0],
            'min_y': min_coords[1], 'max_y': max_coords[1],
            'min_z': min_coords[2], 'max_z': max_coords[2]
        }

    def _check_overlap_xz(self, box_a: Dict, box_b: Dict) -> bool:
        if (box_a['max_x'] < box_b['min_x'] or box_a['min_x'] > box_b['max_x'] or
            box_a['max_z'] < box_b['min_z'] or box_a['min_z'] > box_b['max_z']):
            return False
        return True

    def _align_all_to_common_floor(self, scene_objects: List[Dict]) -> Tuple[List[Dict], float]:
        """Logic rơi tự do và xếp chồng (Stacking)."""
        print("  > Applying gravity and stacking logic...")
        if not scene_objects: return scene_objects, 0.0

        lighting_objs = []
        physics_objs = []
        all_vertices_list = []

        for obj in scene_objects:
            if "Lighting" in obj['label']:
                lighting_objs.append(obj)
            else:
                obj['_aabb'] = self._get_world_aabb(obj['vertices'])
                physics_objs.append(obj)
                all_vertices_list.append(obj['vertices'])

        if not physics_objs: return scene_objects, 0.0

        # Global Floor
        all_v = np.vstack(all_vertices_list)
        global_floor_y = np.min(all_v[:, 1])

        # Sort từ thấp lên cao để xếp nền trước
        physics_objs.sort(key=lambda o: o['_aabb']['min_y'])
        settled_objects = []

        for current_obj in physics_objs:
            current_aabb = current_obj['_aabb']
            target_y = global_floor_y

            # Check va chạm với vật đã xếp
            for settled_obj in settled_objects:
                settled_aabb = settled_obj['_aabb']
                # Nếu vật này ở trên vật kia VÀ có chồng lấn XZ
                if (current_aabb['min_y'] >= settled_aabb['min_y'] - 0.1 and
                    self._check_overlap_xz(current_aabb, settled_aabb)):
                    target_y = max(target_y, settled_aabb['max_y'])

            shift_y = target_y - current_aabb['min_y']
            if abs(shift_y) > 1e-5:
                shift_vec = np.array([0.0, shift_y, 0.0])
                current_obj['vertices'] += shift_vec
                current_obj['center'] = (np.array(current_obj['center']) + shift_vec).tolist()
                current_obj['_aabb']['min_y'] += shift_y
                current_obj['_aabb']['max_y'] += shift_y

            settled_objects.append(current_obj)

        final_objs = lighting_objs + settled_objects
        for obj in final_objs:
            if '_aabb' in obj: del obj['_aabb']

        return final_objs, global_floor_y

    def _realign_objects_to_key(self, objects_data: List[Dict]) -> List[Dict]:
        """Giữ nguyên logic xoay theo vật thể lớn nhất."""
        if not objects_data: return objects_data
        volumes = [obj['bounding_box'][0] * obj['bounding_box'][1] * obj['bounding_box'][2] for obj in objects_data]
        key_idx = np.argmax(volumes)
        key_obj = objects_data[0]
        key_quat = np.quaternion(*key_obj['rotation']).normalized()
        correction_quat = key_quat.inverse()

        aligned_objects = []
        for obj in objects_data:
            curr_quat = np.quaternion(*obj['rotation']).normalized()
            new_quat = correction_quat * curr_quat
            curr_center = np.array(obj['center'])
            new_center = quaternion.rotate_vectors(correction_quat, curr_center)

            new_obj = obj.copy()
            new_obj['rotation'] = [new_quat.w, new_quat.x, new_quat.y, new_quat.z]
            new_obj['center'] = new_center.tolist()
            aligned_objects.append(new_obj)
        return aligned_objects

    def load_scene_from_json(self, json_path: str, obj_folder: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        obj_folder_path = Path(obj_folder)

        for scene_id, raw_objects in data.items():
            print(f"\nLoading scene: {scene_id}")
            aligned_metadata = self._realign_objects_to_key(raw_objects)
            scene_objects = []

            # Để gộp mtl, ta tạo list chứa các dòng text của mtl
            scene_mtl_lines = []

            for obj_data in aligned_metadata:
                label = obj_data['label']
                obj_file = obj_folder_path / f"{label}.obj"
                if not obj_file.exists(): continue

                try:
                    # Load Full (V, VT, VN, MTL)
                    obj_full = self.load_obj_with_mtl(str(obj_file), label)

                    # Transform (với pre-rotation được áp dụng bên trong)
                    self.transform_object(
                        obj_full,
                        obj_data['bounding_box'],
                        np.array(obj_data['rotation']),
                        np.array(obj_data['center'])
                    )

                    # Thêm metadata cần thiết
                    obj_full['label'] = label
                    obj_full['rotation'] = obj_data['rotation']
                    obj_full['center'] = obj_data['center']
                    obj_full['bounding_box'] = obj_data['bounding_box']

                    scene_objects.append(obj_full)
                    scene_mtl_lines.extend(obj_full['materials_content'])

                except Exception as e:
                    print(f"  Error loading {label}: {e}")

            # Stacking / Gravity
            scene_objects, floor_y = self._align_all_to_common_floor(scene_objects)

            self.scenes[scene_id] = scene_objects
            self.scene_floor_heights[scene_id] = floor_y
            self.scene_materials[scene_id] = scene_mtl_lines

            print(f"Scene {scene_id}: Loaded {len(scene_objects)} objects.")

    def export_scene(self, scene_id: str, output_path: str, include_floor: bool = True):
        if scene_id not in self.scenes: return
        objects = self.scenes[scene_id]
        floor_height = self.scene_floor_heights.get(scene_id, 0.0)
        mtl_lines = self.scene_materials.get(scene_id, [])

        out_obj_path = Path(output_path)
        out_mtl_path = out_obj_path.with_suffix('.mtl')
        mtl_filename = out_mtl_path.name

        # 1. Ghi file .mtl tổng hợp
        with open(out_mtl_path, 'w') as f_mtl:
            f_mtl.write(f"# Material Lib for Scene {scene_id}\n")
            for line in mtl_lines:
                f_mtl.write(line + "\n")

            # Thêm material cho sàn (Floor)
            if include_floor:
                f_mtl.write("\nnewmtl Floor_Mat\n")
                f_mtl.write("Kd 0.8 0.8 0.8\n") # Màu xám sáng
                f_mtl.write("Ns 10.0\n")

        # 2. Ghi file .obj
        with open(out_obj_path, 'w') as f:
            f.write(f"# 3D Scene: {scene_id}\n")
            f.write(f"mtllib {mtl_filename}\n") # Link tới file mtl vừa tạo

            # Offsets (OBJ index bắt đầu từ 1, và tăng dần qua các object)
            v_offset = 0
            vt_offset = 0
            vn_offset = 0

            # --- Sàn (Floor) ---
            if include_floor:
                f.write(f"o Floor\n")
                w, d = self.floor_size[0]/2, self.floor_size[1]/2
                y = floor_height
                # 4 đỉnh sàn
                f.write(f"v {-w:.4f} {y:.4f} {-d:.4f}\n")
                f.write(f"v {w:.4f} {y:.4f} {-d:.4f}\n")
                f.write(f"v {w:.4f} {y:.4f} {d:.4f}\n")
                f.write(f"v {-w:.4f} {y:.4f} {d:.4f}\n")
                # 1 normal hướng lên trên
                f.write("vn 0.0 1.0 0.0\n")
                f.write("usemtl Floor_Mat\n")
                # Face: v/vt/vn (ở đây ko có vt, dùng v//vn)
                # Normal index là 1 (vì chưa có obj nào trước)
                f.write(f"f {1+v_offset}//{1+vn_offset} {2+v_offset}//{1+vn_offset} {3+v_offset}//{1+vn_offset} {4+v_offset}//{1+vn_offset}\n\n")

                v_offset += 4
                vn_offset += 1

            # --- Các Objects ---
            for obj in objects:
                f.write(f"o {obj['label']}\n")

                # Write Vertices
                for v in obj['vertices']:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                # Write UVs
                if obj['uvs'] is not None:
                    for vt in obj['uvs']:
                        f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")

                # Write Normals
                if obj['normals'] is not None:
                    for vn in obj['normals']:
                        f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")

                current_w_mat = None
                for face_data in obj['faces']:
                    mat_name = face_data['material']
                    if mat_name and mat_name != current_w_mat:
                        f.write(f"usemtl {mat_name}\n")
                        current_w_mat = mat_name

                    face_str_parts = []
                    for idx_tuple in face_data['indices']:
                        v_idx, vt_idx, vn_idx = idx_tuple

                        # Tính global index
                        s_v = str(v_idx + v_offset + 1)
                        s_vt = str(vt_idx + vt_offset + 1) if vt_idx is not None else ""
                        s_vn = str(vn_idx + vn_offset + 1) if vn_idx is not None else ""

                        # Tạo chuỗi v/vt/vn
                        if not s_vt and not s_vn:
                            face_str_parts.append(s_v)
                        elif not s_vt and s_vn:
                            face_str_parts.append(f"{s_v}//{s_vn}")
                        else:
                            face_str_parts.append(f"{s_v}/{s_vt}/{s_vn}")

                    f.write(f"f {' '.join(face_str_parts)}\n")

                # Cập nhật Offset cho object tiếp theo
                v_offset += len(obj['vertices'])
                if obj['uvs'] is not None: vt_offset += len(obj['uvs'])
                if obj['normals'] is not None: vn_offset += len(obj['normals'])

                f.write("\n")

        print(f"✓ Exported: {output_path} (with MTL)")

    def export_all_scenes(self, output_folder: str, include_floor: bool = True):
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        for scene_id in self.scenes.keys():
            safe_name = scene_id.replace('/', '_').replace('\\', '_')
            self.export_scene(scene_id, str(output_path / f"{safe_name}.obj"), include_floor)

if __name__ == "__main__":
    # Update đường dẫn phù hợp với môi trường của bạn
    orchestrator = SceneOrchestrator()
    orchestrator.load_scene_from_json(
        "/content/pos.json", # File JSON input
        "/content/test"      # Folder chứa .obj và .mtl
    )
    orchestrator.export_all_scenes("output_scenes_full")