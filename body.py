import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
import contextlib
import cv2
import mediapipe as mp
import numpy as np
import glob
from tqdm import tqdm
import warnings
from sklearn.cluster import DBSCAN
try:
    from mediapipe.python.solutions import pose as mp_pose  # type: ignore
except Exception:  # pragma: no cover
    mp_pose = None
try:
    from mediapipe.python.solutions import drawing_utils as mp_drawing  # type: ignore
except Exception:  # pragma: no cover
    mp_drawing = None
import shared as shared_module
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
@contextlib.contextmanager

def _suppress_native_stderr():
    try:
        null = open(os.devnull, "w")
    except Exception:
        yield
        return
    try:
        old_fd = os.dup(2)
    except Exception:
        try:
            yield
        finally:
            try:
                null.close()
            except Exception:
                pass
        return
    try:
        os.dup2(null.fileno(), 2)
        yield
    finally:
        try:
            os.dup2(old_fd, 2)
        except Exception:
            pass
        try:
            os.close(old_fd)
        except Exception:
            pass
        try:
            null.close()
        except Exception:
            pass


class BodyReshaper:
    def __init__(self, db_eps=50.0, db_minpts=2):
        self.db_eps = db_eps
        self.db_minpts = db_minpts

        self.mp_pose = mp_pose if mp_pose is not None else getattr(mp.solutions, "pose")
        with _suppress_native_stderr():
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )

    # dlib(face_recognition) landmarks
    def get_face_landmarks(self, image):
        fr = shared_module._try_import_face_recognition()
        if fr is None:
            return None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lm_list = fr.face_landmarks(rgb)
        if not lm_list:
            return None
        return lm_list[0]

    def reshape_face_only(self, image, mode="slim", face_landmarks=None):

        if face_landmarks is None:
            face_landmarks = self.get_face_landmarks(image)
        if not face_landmarks:
            return self.reshape_character(image, mode=mode)

        chin = face_landmarks.get("chin")
        nose_bridge = face_landmarks.get("nose_bridge")
        if not chin or not nose_bridge:
            return image

        chin_pts = np.array(chin, dtype=np.float32)
        face_width = float(np.linalg.norm(chin_pts[0] - chin_pts[-1]))
        if face_width < 20:
            return image

        center = np.mean(np.array(nose_bridge, dtype=np.float32), axis=0)
        center = (int(center[0]), int(center[1]))

        left_cheek = (int(chin_pts[4][0]), int(chin_pts[4][1]))
        right_cheek = (int(chin_pts[12][0]), int(chin_pts[12][1]))
        jaw = (int(chin_pts[8][0]), int(chin_pts[8][1]))

        result = image.copy()
        if mode == "slim":
            cheek_strength = -0.10
            jaw_strength = -0.08
        else:
            cheek_strength = 0.14
            jaw_strength = 0.12

        cheek_radius = int(face_width * 0.28)
        jaw_radius = int(face_width * 0.34)
        center_radius = int(face_width * 0.26)


        result = self.apply_liquify(result, left_cheek, cheek_radius, cheek_strength)
        result = self.apply_liquify(result, right_cheek, cheek_radius, cheek_strength)
        result = self.apply_liquify(result, jaw, jaw_radius, jaw_strength)

        center_strength = (-0.06 if mode == "slim" else 0.08)
        result = self.apply_liquify(result, center, center_radius, center_strength)
        return result

    # Pose landmark detection
    def get_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with _suppress_native_stderr():
            results = self.pose.process(image_rgb)
        pose_landmarks = getattr(results, "pose_landmarks", None)
        if not pose_landmarks:
            return None

        h, w, _ = image.shape
        landmarks = {}
        for idx, landmark in enumerate(pose_landmarks.landmark):
            landmarks[idx] = (int(landmark.x * w), int(landmark.y * h))
        return landmarks

    # Local liquify warp (radial, smooth falloff)
    def apply_liquify(self, image, center, radius, strength):
        h, w = image.shape[:2]
        cx, cy = center
        if radius <= 1:
            return image

        pad = int(radius * 0.2)
        x1 = max(0, cx - radius - pad)
        x2 = min(w, cx + radius + pad)
        y1 = max(0, cy - radius - pad)
        y2 = min(h, cy + radius + pad)

        if x1 >= x2 or y1 >= y2:
            return image

        roi = image[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]

        grid_x, grid_y = np.meshgrid(np.arange(rw), np.arange(rh))

        cx_local = cx - x1
        cy_local = cy - y1

        dx = grid_x - cx_local
        dy = grid_y - cy_local
        dist_sq = dx * dx + dy * dy
        radius_sq = float(radius * radius)

        inside = dist_sq < radius_sq
        if not np.any(inside):
            return image

        dist_sq_norm = np.zeros_like(dist_sq, dtype=np.float32)
        dist_sq_norm[inside] = dist_sq[inside] / radius_sq
        factor = np.zeros_like(dist_sq_norm, dtype=np.float32)
        factor[inside] = (1.0 - dist_sq_norm[inside]) ** 2

        deform = factor * strength
        map_x = (grid_x - dx * deform).astype(np.float32)
        map_y = (grid_y - dy * deform).astype(np.float32)

        warped_roi = cv2.remap(
            roi, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        result = image.copy()
        result[y1:y2, x1:x2] = warped_roi
        return result
    # Warp along a limb segment
    def process_segment(self, image, p1, p2, strength, radius_factor=0.5, steps=5):
        vec = np.array(p2) - np.array(p1)
        length = np.linalg.norm(vec)
        if length == 0:
            return image

        radius = int(length * radius_factor)
        radius = max(radius, 2)

        for t in np.linspace(0.1, 0.9, steps):
            center = (int(p1[0] + vec[0] * t), int(p1[1] + vec[1] * t))
            image = self.apply_liquify(image, center, radius, strength)

        return image

    # Global body warp (overall slim/fat)
    def global_body_warp(self, image, landmarks, mode='slim', strength_scale=1.0):
        key_ids = [0, 7, 8, 11, 12, 23, 24, 25, 26, 27, 28]
        xs, ys = [], []
        for k in key_ids:
            if k in landmarks:
                x, y = landmarks[k]
                xs.append(x)
                ys.append(y)

        if len(xs) < 3:
            return image

        xs = np.array(xs)
        ys = np.array(ys)

        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        h, w = image.shape[:2]
        margin_x = int((max_x - min_x) * 0.25)
        margin_y = int((max_y - min_y) * 0.25)

        min_x = max(0, min_x - margin_x)
        max_x = min(w - 1, max_x + margin_x)
        min_y = max(0, min_y - margin_y)
        max_y = min(h - 1, max_y + margin_y)

        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2 + int(0.1 * (max_y - min_y))

        box_w = max_x - min_x
        box_h = max_y - min_y
        radius = int(max(box_w, box_h) * 0.7)
        radius = max(radius, 50)

        if mode == 'slim':
            strength = -0.06 * strength_scale
        else:
            strength = 0.09 * strength_scale

        return self.apply_liquify(image, (cx, cy), radius, strength)

    def shrink_head(self, image, landmarks, scale: float = 0.92):
        if 0 not in landmarks or 7 not in landmarks or 8 not in landmarks:
            return image

        h, w = image.shape[:2]
        nose = landmarks[0]
        l_ear = landmarks[7]
        r_ear = landmarks[8]

        face_width = float(np.linalg.norm(np.array(l_ear) - np.array(r_ear)))
        if face_width < 10:
            return image

        top = int(nose[1] - 1.3 * face_width)
        bottom = int(nose[1] + 1.0 * face_width)
        left = int(nose[0] - 1.2 * face_width)
        right = int(nose[0] + 1.2 * face_width)

        top = max(0, top)
        bottom = min(h - 1, bottom)
        left = max(0, left)
        right = min(w - 1, right)

        if bottom <= top + 4 or right <= left + 4:
            return image

        roi = image[top:bottom, left:right]
        rh, rw = roi.shape[:2]

        grid_x, grid_y = np.meshgrid(np.arange(rw), np.arange(rh))

        cx_local = nose[0] - left
        cy_local = nose[1] - top + int(0.1 * face_width)

        rx = rw / 2.0
        ry = rh / 2.0

        dx = grid_x - cx_local
        dy = grid_y - cy_local
        dist_norm = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry)

        inside = dist_norm < 1.0
        if not np.any(inside):
            return image

        global_y = grid_y + top
        chest_limit = nose[1] + int(1.1 * face_width)
        inside &= (global_y < chest_limit)
        if not np.any(inside):
            return image

        falloff = np.zeros_like(dist_norm, dtype=np.float32)
        falloff[inside] = (1.0 - dist_norm[inside]) ** 2

        scale = float(np.clip(scale, 0.85, 1.0))
        x_scaled = cx_local + (grid_x - cx_local) * scale
        y_scaled = cy_local + (grid_y - cy_local) * scale

        map_x = grid_x.astype(np.float32)
        map_y = grid_y.astype(np.float32)
        map_x[inside] = grid_x[inside] + (x_scaled[inside] - grid_x[inside]) * falloff[inside]
        map_y[inside] = grid_y[inside] + (y_scaled[inside] - grid_y[inside]) * falloff[inside]

        warped_roi = cv2.remap(
            roi,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        result = image.copy()
        result[top:bottom, left:right] = warped_roi
        return result

    def shrink_face_radial(self, image, landmarks, scale: float = 0.93):
        if 0 not in landmarks or 7 not in landmarks or 8 not in landmarks:
            return image

        h, w = image.shape[:2]
        nose = landmarks[0]
        l_ear = landmarks[7]
        r_ear = landmarks[8]

        face_width = np.linalg.norm(np.array(l_ear) - np.array(r_ear))
        if face_width < 10:
            return image

        cx = nose[0]
        cy = nose[1] - int(0.05 * face_width)

        radius = int(face_width * 0.95)
        radius = min(radius, cx, cy, w - 1 - cx, h - 1 - cy)
        if radius <= 5:
            return image

        scale = float(np.clip(scale, 0.85, 1.0))
        strength = -0.20 * (1.0 - scale) / (1.0 - 0.85 + 1e-6)

        result = self.apply_liquify(
            image,
            center=(cx, cy),
            radius=radius,
            strength=strength
        )
        return result
    def widen_face_horizontal(self, image, landmarks, amount: float = 0.12):
        if 0 not in landmarks or 7 not in landmarks or 8 not in landmarks:
            return image

        h, w = image.shape[:2]
        nose = landmarks[0]
        l_ear = landmarks[7]
        r_ear = landmarks[8]

        face_width = float(np.linalg.norm(np.array(l_ear) - np.array(r_ear)))
        if face_width < 10:
            return image

        top = int(nose[1] - 0.8 * face_width)
        bottom = int(nose[1] + 0.9 * face_width)
        left = int(nose[0] - 1.0 * face_width)
        right = int(nose[0] + 1.0 * face_width)

        top = max(0, top)
        bottom = min(h - 1, bottom)
        left = max(0, left)
        right = min(w - 1, right)

        if bottom <= top + 4 or right <= left + 4:
            return image

        roi = image[top:bottom, left:right]
        rh, rw = roi.shape[:2]

        grid_x, grid_y = np.meshgrid(np.arange(rw), np.arange(rh))

        cx_local = nose[0] - left
        cy_local = nose[1] - top

        rx = rw * 0.45
        ry = rh * 0.55

        dx = grid_x - cx_local
        dy = grid_y - cy_local
        dist_norm = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry)

        inside = dist_norm < 1.0
        if not np.any(inside):
            return image

        falloff = np.zeros_like(dist_norm, dtype=np.float32)
        falloff[inside] = (1.0 - dist_norm[inside]) ** 2

        global_y = grid_y + top
        mouth_y = nose[1] + int(0.3 * face_width)
        mouth_band_half = int(0.20 * face_width)
        mouth_band_low = mouth_y - mouth_band_half
        mouth_band_high = mouth_y + mouth_band_half
        mouth_region = (global_y >= mouth_band_low) & (global_y <= mouth_band_high)
        reduce_mask = inside & mouth_region
        falloff[reduce_mask] *= 0.8

        amount = float(amount)
        scale_x = 1.0 + amount * falloff
        scale_y = 1.0 + 0.3 * amount * falloff

        new_x = cx_local + dx * scale_x
        new_y = cy_local + dy * scale_y

        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)

        warped_roi = cv2.remap(
            roi,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        result = image.copy()
        result[top:bottom, left:right] = warped_roi
        return result
    def preprocess_text_image(self, img):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        binary_fg = 255 - binary

        kernel = np.ones((3, 3), np.uint8)
        binary_fg = cv2.morphologyEx(
            binary_fg, cv2.MORPH_OPEN, kernel, iterations=1
        )
        return binary_fg

    def extract_points_text(self, binary_fg):
        ys, xs = np.where(binary_fg > 0)
        if len(ys) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.vstack([ys, xs]).T.astype(np.float32)

    def apply_dbscan_text(self, points):
        if len(points) == 0:
            return np.array([], dtype=int)
        return DBSCAN(
            eps=self.db_eps,
            min_samples=self.db_minpts
        ).fit_predict(points)

    def dbscan_morph_text(self, binary_fg, points, labels, mode="fat", ksize=3):
        h, w = binary_fg.shape[:2]
        result = np.zeros_like(binary_fg)

        unique_labels = [lb for lb in np.unique(labels) if lb != -1]
        kernel = np.ones((ksize, ksize), np.uint8)

        for lbl in unique_labels:
            cluster_mask = np.zeros((h, w), dtype=np.uint8)
            pts = points[labels == lbl].astype(int)
            ys = np.clip(pts[:, 0], 0, h - 1)
            xs = np.clip(pts[:, 1], 0, w - 1)
            cluster_mask[ys, xs] = 255

            if mode == "fat":
                proc = cv2.dilate(cluster_mask, kernel, iterations=1)
            else:
                proc = cv2.erode(cluster_mask, kernel, iterations=1)

            result = cv2.bitwise_or(result, proc)

        if mode == "fat":
            result = cv2.bitwise_or(result, binary_fg)
        else:
            result = cv2.bitwise_and(result, binary_fg)

        return result

    def reshape_character(self, image, mode="slim"):
        binary_fg = self.preprocess_text_image(image)
        points = self.extract_points_text(binary_fg)

        if points.shape[0] == 0:
            return image

        labels = self.apply_dbscan_text(points)

        eps_ref = 16.0
        minpts_ref = 30.0
        strength = 0.5 * (self.db_eps / eps_ref) + 0.5 * (self.db_minpts / minpts_ref)
        strength = np.clip(strength, 0.3, 3.0)

        k = 1 + int(strength * 6)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1

        out_fg = self.dbscan_morph_text(
            binary_fg,
            points,
            labels,
            mode=("fat" if mode == "fat" else "slim"),
            ksize=k
        )
        out_show = 255 - out_fg
        out_bgr = cv2.cvtColor(out_show, cv2.COLOR_GRAY2BGR)
        return out_bgr
    def reshape_body(self, image, mode='slim'):
        landmarks = self.get_landmarks(image)
        if not landmarks:
            face_lm = self.get_face_landmarks(image)
            if face_lm:
                return self.reshape_face_only(image, mode=mode, face_landmarks=face_lm)
            return self.reshape_character(image, mode=mode)

        result = image.copy()

        keys = list(landmarks.keys())
        coords = np.array([landmarks[k] for k in keys], dtype=np.float32)

        ref_eps = 50.0
        ref_minpts = 5.0
        factor_eps = np.clip(self.db_eps / ref_eps, 0.5, 2.0)
        factor_minpts = np.clip(self.db_minpts / ref_minpts, 0.5, 2.0)
        strength_scale = 0.5 * factor_eps + 0.5 * factor_minpts

        cluster_labels = {}
        main_cluster_label = None

        if len(coords) >= 2:
            clustering = DBSCAN(
                eps=self.db_eps,
                min_samples=self.db_minpts
            ).fit(coords)
            labels = clustering.labels_

            unique, counts = np.unique(labels, return_counts=True)
            cluster_sizes = {int(l): int(c) for l, c in zip(unique, counts) if l != -1}
            if cluster_sizes:
                main_cluster_label = max(cluster_sizes, key=lambda k: cluster_sizes[k])

            for i, k in enumerate(keys):
                cluster_labels[k] = int(labels[i])
        else:
            for k in keys:
                cluster_labels[k] = 0
            main_cluster_label = 0

        limbs = [
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (23, 25), (25, 27),
            (24, 26), (26, 28)
        ]

        base_limb_strength = (-0.10 if mode == 'slim' else 0.15) * strength_scale
        limb_radius = 0.3 if mode == 'slim' else 0.4

        def get_limb_strength(joint_idx):
            lbl = cluster_labels.get(joint_idx, -1)
            if lbl == main_cluster_label and lbl is not None:
                return base_limb_strength * 1.2
            else:
                return base_limb_strength * 0.8

        for start, end in limbs:
            if start in landmarks and end in landmarks:
                strength = get_limb_strength(start)
                result = self.process_segment(
                    result,
                    landmarks[start],
                    landmarks[end],
                    strength=strength,
                    radius_factor=limb_radius,
                    steps=4
                )

        if 11 in landmarks and 12 in landmarks and 23 in landmarks and 24 in landmarks:
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            l_hip = landmarks[23]
            r_hip = landmarks[24]

            torso_width = np.linalg.norm(np.array(l_hip) - np.array(r_hip))

            if mode == 'slim':
                torso_strength_side = base_limb_strength
                result = self.process_segment(result, l_shoulder, l_hip, torso_strength_side, 0.6, 4)
                result = self.process_segment(result, r_shoulder, r_hip, torso_strength_side, 0.6, 4)
            else:
                belly_center = (
                    (l_hip[0] + r_hip[0] + l_shoulder[0] + r_shoulder[0]) // 4,
                    (l_hip[1] + r_hip[1] + l_shoulder[1] + r_shoulder[1]) // 4
                    + int(torso_width * 0.2)
                )
                belly_strength = 0.25 * strength_scale
                result = self.apply_liquify(result, belly_center, int(torso_width * 1.2), belly_strength)

                side_strength = 0.1 * strength_scale
                result = self.process_segment(result, l_shoulder, l_hip, side_strength, 0.5, 3)
                result = self.process_segment(result, r_shoulder, r_hip, side_strength, 0.5, 3)
        if 0 in landmarks and 7 in landmarks and 8 in landmarks:
            nose = landmarks[0]
            l_ear = landmarks[7]
            r_ear = landmarks[8]
            face_width = np.linalg.norm(np.array(l_ear) - np.array(r_ear))
            l_cheek = ((nose[0] + l_ear[0]) // 2, (nose[1] + l_ear[1]) // 2)
            r_cheek = ((nose[0] + r_ear[0]) // 2, (nose[1] + r_ear[1]) // 2)
            chin = (nose[0], nose[1] + int(face_width * 0.6))
            if mode == 'slim':
                cheek_strength = -0.06 * strength_scale
                chin_strength = -0.08 * strength_scale
                cheek_radius = int(face_width * 0.45)
                chin_radius = int(face_width * 0.60)
                result = self.apply_liquify(result, l_cheek, cheek_radius, cheek_strength)
                result = self.apply_liquify(result, r_cheek, cheek_radius, cheek_strength)
                result = self.apply_liquify(result, chin, chin_radius, chin_strength)
                jaw_center = (nose[0], nose[1] + int(face_width * 0.75))
                jaw_radius = int(face_width * 0.65)
                jaw_strength = -0.06 * strength_scale
                result = self.apply_liquify(result, jaw_center, jaw_radius, jaw_strength)
            else:
                cheek_strength = 0.15 * strength_scale
                result = self.apply_liquify(result, l_cheek, int(face_width * 0.4), cheek_strength)
                result = self.apply_liquify(result, r_cheek, int(face_width * 0.4), cheek_strength)
                double_chin = (nose[0], nose[1] + int(face_width * 0.8))
                result = self.apply_liquify(result, double_chin, int(face_width * 0.6), 0.15 * strength_scale)
                if 11 in landmarks and 12 in landmarks:
                    l_neck = (
                        (l_ear[0] + landmarks[11][0]) // 2,
                        (l_ear[1] + landmarks[11][1]) // 2
                    )
                    r_neck = (
                        (r_ear[0] + landmarks[12][0]) // 2,
                        (r_ear[1] + landmarks[12][1]) // 2
                    )
                    result = self.apply_liquify(result, l_neck, int(face_width * 0.5), 0.1 * strength_scale)
                    result = self.apply_liquify(result, r_neck, int(face_width * 0.5), 0.1 * strength_scale)

        if (
            mode == "slim"
            and 11 in landmarks
            and 12 in landmarks
            and 0 in landmarks
            and 7 in landmarks
            and 8 in landmarks
        ):
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            shoulder_width = np.linalg.norm(
                np.array(l_shoulder) - np.array(r_shoulder)
            )
            nose = landmarks[0]
            l_ear = landmarks[7]
            r_ear = landmarks[8]
            face_width = np.linalg.norm(np.array(l_ear) - np.array(r_ear))
            safe_top = nose[1] + int(0.2 * face_width)
            neck_center = (
                (l_shoulder[0] + r_shoulder[0]) // 2,
                (l_shoulder[1] + r_shoulder[1]) // 2 - int(0.05 * shoulder_width)
            )
            neck_radius = int(0.40 * shoulder_width)
            top_of_effect = neck_center[1] - neck_radius
            if top_of_effect < safe_top:
                shift = safe_top - top_of_effect
                neck_center = (neck_center[0], neck_center[1] + shift)
            neck_strength = -0.15 * strength_scale
            result = self.apply_liquify(result, neck_center, neck_radius, neck_strength)
        result = self.global_body_warp(result, landmarks, mode=mode, strength_scale=strength_scale)
        if mode == "slim":
            result = self.shrink_head(result, landmarks, scale=0.92)
        else:
            result = self.widen_face_horizontal(result, landmarks, amount=0.12)
        return result


def export_body_analysis_image(image):
    pose_module = mp_pose if mp_pose is not None else getattr(mp.solutions, "pose")
    drawing = mp_drawing if mp_drawing is not None else getattr(mp.solutions, "drawing_utils", None)
    if drawing is None:
        return image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with _suppress_native_stderr():
        with pose_module.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            results = pose.process(rgb)
            pose_landmarks = getattr(results, "pose_landmarks", None)
            if not pose_landmarks:
                return image
            out = image.copy()
            drawing.draw_landmarks(
                out,
                pose_landmarks,
                list(pose_module.POSE_CONNECTIONS),
            )
            return out


def run_body_analysis_on_dir(input_dir: str, output_dir: str, recursive: bool):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    image_paths: list[str] = []
    if recursive:
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    else:
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    if not image_paths:
        print("[WARNING] No images found for body analysis.")
        return
    for img_path in tqdm(image_paths, desc="Body Analysis", unit="img"):
        image = cv2.imread(img_path)
        if image is None:
            continue
        rel_dir = os.path.relpath(os.path.dirname(img_path), input_dir)
        out_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        out_img = export_body_analysis_image(image)
        cv2.imwrite(os.path.join(out_dir, f"{name}_body_analysis{ext}"), out_img)


def run_body_reshape_best_on_dir(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    slim_eps: float,
    slim_minpts: int,
    fat_eps: float,
    fat_minpts: int,
    export_mode: str = "both",
):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    image_paths: list[str] = []
    if recursive:
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    else:
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    if not image_paths:
        print("[WARNING] No images found for reshape_best.")
        return
    reshaper = BodyReshaper()
    for img_path in tqdm(image_paths, desc="Reshape Best", unit="img"):
        image = cv2.imread(img_path)
        if image is None:
            continue
        rel_dir = os.path.relpath(os.path.dirname(img_path), input_dir)
        out_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        if export_mode in ["slim", "both"]:
            reshaper.db_eps = float(slim_eps)
            reshaper.db_minpts = int(slim_minpts)
            out = reshaper.reshape_body(image, mode="slim")
            cv2.imwrite(os.path.join(out_dir, f"{name}_slim_best{ext}"), out)
        if export_mode in ["fat", "both"]:
            reshaper.db_eps = float(fat_eps)
            reshaper.db_minpts = int(fat_minpts)
            out = reshaper.reshape_body(image, mode="fat")
            cv2.imwrite(os.path.join(out_dir, f"{name}_fat_best{ext}"), out)


# # Build eps x MinPts grid image for one mode
def build_param_grid(image, reshaper, eps_list, minpts_list, mode="slim"):
    h, w = image.shape[:2]
    rows = len(eps_list) + 1
    cols = len(minpts_list) + 1
    grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255
    grid[0:h, 0:w] = image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, h / 500.0 * 1.2)
    thickness = 1
    text_color = (0, 0, 0)
    def put_center_text(canvas, text):
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = (canvas.shape[1] - tw) // 2
        y = (canvas.shape[0] + th) // 2
        cv2.putText(canvas, text, (x, y),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)
    for j, minpts in enumerate(minpts_list):
        label_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        txt = f"MinPts = {minpts}"
        put_center_text(label_img, txt)
        grid[0:h, (j + 1) * w:(j + 2) * w] = label_img
    for i, eps in enumerate(eps_list):
        label_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        txt = f"eps = {eps:.1f}"
        put_center_text(label_img, txt)
        grid[(i + 1) * h:(i + 2) * h, 0:w] = label_img
        for j, minpts in enumerate(minpts_list):
            reshaper.db_eps = eps
            reshaper.db_minpts = minpts
            out = reshaper.reshape_body(image, mode=mode)
            grid[
                (i + 1) * h:(i + 2) * h,
                (j + 1) * w:(j + 2) * w
            ] = out
    return grid
