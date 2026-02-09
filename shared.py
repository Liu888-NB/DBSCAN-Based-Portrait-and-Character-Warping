import cv2
import numpy as np
import os
import glob
import sys
from tqdm import tqdm
import warnings
from sklearn.cluster import DBSCAN
import pickle
from typing import cast

try:
    from imutils import paths
    from imutils import build_montages
except Exception:  # pragma: no cover
    paths = None
    build_montages = None

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

face_recognition = None
_face_import_error: Exception | None = None

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

if os.name == "nt":
    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    dll_dir = os.path.join(conda_prefix, "Library", "bin")
    if os.path.isdir(dll_dir):
        os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(dll_dir)  # type: ignore[attr-defined]
        except Exception:
            pass


def _try_import_face_recognition():
    global face_recognition, _face_import_error
    if face_recognition is not None:
        return face_recognition
    try:
        import face_recognition as fr  # dlib-based

        face_recognition = fr
        _face_import_error = None
        return face_recognition
    except Exception as e:
        _face_import_error = e
        return None


def _require_face_recognition():
    fr = _try_import_face_recognition()
    if fr is None:
        detail = f"\nUnderlying error: {_face_import_error}" if _face_import_error is not None else ""
        raise RuntimeError(
            "Failed to import face_recognition/dlib: The current environment cannot use the dlib face module.\n"
            "You can continue to use the Mediapipe body reshaping features of body_reshape_mix_v5;\n"
            "To enable dlib+DBSCAN face recognition/clustering/morphing, please fix the face_recognition dependencies."
            + detail
        )


def encode_faces(photos_dir, encodings_path, detection_method="hog", max_detect_size: int = 800):
    _require_face_recognition()
    fr = cast(object, face_recognition)
    if paths is None:
        raise RuntimeError("Missing imutils: unable to scan image directory. Please install imutils.")

    print("[INFO] Starting to encode faces...")
    image_paths = list(paths.list_images(photos_dir))
    if not image_paths:
        print(f"[WARNING] No image files found in directory '{photos_dir}'.")
        return
    data = []
    for image_path in tqdm(image_paths, desc="Encoding Faces", unit="img"):
        image = cv2.imread(image_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_small = rgb
        scale = 1.0
        if detection_method == "cnn" and max_detect_size and max_detect_size > 0:
            h0, w0 = rgb.shape[:2]
            max_dim = max(h0, w0)
            if max_dim > max_detect_size:
                scale = max_detect_size / float(max_dim)
                new_w = max(2, int(w0 * scale))
                new_h = max(2, int(h0 * scale))
                rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        boxes = []
        encodings = []
        retry_scales = [1.0]
        if detection_method == "cnn":
            retry_scales = [1.0, 0.85, 0.7, 0.55, 0.4]
        last_err = None
        for rs in retry_scales:
            try:
                if rs != 1.0:
                    h1, w1 = rgb_small.shape[:2]
                    rgb_try = cv2.resize(
                        rgb_small,
                        (max(2, int(w1 * rs)), max(2, int(h1 * rs))),
                        interpolation=cv2.INTER_AREA,
                    )
                    scale_try = scale * rs
                else:
                    rgb_try = rgb_small
                    scale_try = scale
                boxes_try = cast("object", fr).face_locations(rgb_try, model=detection_method)  # type: ignore[attr-defined]
                enc_try = cast("object", fr).face_encodings(rgb_try, boxes_try)  # type: ignore[attr-defined]
                if scale_try != 1.0:
                    mapped = []
                    for (top, right, bottom, left) in boxes_try:
                        mapped.append(
                            (
                                int(top / scale_try),
                                int(right / scale_try),
                                int(bottom / scale_try),
                                int(left / scale_try),
                            )
                        )
                    boxes = mapped
                else:
                    boxes = boxes_try
                encodings = enc_try
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                continue
        if last_err is not None:
            print(f"[WARNING] face_locations(cnn) failed on {os.path.basename(image_path)}: {last_err}")
            continue
        d = [
            {"imagePath": image_path, "loc": box, "encoding": enc}
            for (box, enc) in zip(boxes, encodings)
        ]
        data.extend(d)
    print("[INFO] Serializing encodings...")
    os.makedirs(os.path.dirname(encodings_path), exist_ok=True)
    with open(encodings_path, "wb") as f:
        f.write(pickle.dumps(data))
    print(f"[INFO] Encodings saved to  {encodings_path}")


def cluster_faces(encodings_path, results_dir, eps=0.5, min_samples=1):
    if build_montages is None:
        raise RuntimeError("lack of imutils: cannot generate montage. Please install imutils.")
    print("[INFO] Loading encodings...")
    try:
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Encoding file not found: {encodings_path}")
        print("Please run encode_faces first to generate the encoding file.")
        return
    encodings = np.array([d["encoding"] for d in data], dtype=np.float32)
    if len(encodings) == 0:
        print("[WARNING] No face encodings found in the encoding file.")
        return
    print("[INFO] clustering faces...")
    clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=float(eps), min_samples=int(min_samples))
    clt.fit(encodings)
    label_ids = np.unique(clt.labels_)
    num_unique_faces = len(np.where(label_ids > -1)[0])
    print(f"[INFO] Found unique faces: {num_unique_faces}")
    clusters_root = os.path.join(results_dir, "Clusters")
    montages_dir = os.path.join(clusters_root, "Montages")
    os.makedirs(montages_dir, exist_ok=True)
    for label_id in tqdm(label_ids, desc="Clustering Faces", unit="cluster"):
        cluster_dir = os.path.join(clusters_root, f"Cluster_{label_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        idxs = np.where(clt.labels_ == label_id)[0]
        faces = []
        for i in idxs:
            image_path = data[i]["imagePath"]
            image = cv2.imread(image_path)
            if image is None:
                continue
            filename = os.path.basename(image_path)
            cv2.imwrite(os.path.join(cluster_dir, filename), image)
            (top, right, bottom, left) = data[i]["loc"]
            face = image[top:bottom, left:right]
            if face.size == 0:
                continue
            face = cv2.resize(face, (96, 96))
            faces.append(face)
        if len(faces) == 0:
            continue
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        cv2.imwrite(os.path.join(montages_dir, f"cluster_{label_id}.jpg"), montage)


def analyze_face_features(image_path, results_dir):
    _require_face_recognition()
    fr = cast(object, face_recognition)
    if KMeans is None:
        raise RuntimeError("lack of sklearn")
    image = cv2.imread(image_path)
    if image is None:
        return
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb_image.shape
    face_landmarks_list = cast("object", fr).face_landmarks(rgb_image)  # type: ignore[attr-defined]
    if not face_landmarks_list:
        return
    landmarks = face_landmarks_list[0]
    heatmap = np.zeros_like(image)
    chin = landmarks["chin"]
    left_eyebrow = landmarks["left_eyebrow"]
    right_eyebrow = landmarks["right_eyebrow"]
    face_contour = chin + [right_eyebrow[-1], right_eyebrow[0], left_eyebrow[-1], left_eyebrow[0]]
    face_contour = np.array(face_contour, dtype=np.int32)
    cv2.fillPoly(heatmap, [face_contour], (153, 204, 255))
    cv2.fillPoly(heatmap, [np.array(landmarks["left_eyebrow"])], (0, 128, 255))
    cv2.fillPoly(heatmap, [np.array(landmarks["right_eyebrow"])], (0, 128, 255))
    cv2.fillPoly(heatmap, [np.array(landmarks["left_eye"])], (0, 255, 0))
    cv2.fillPoly(heatmap, [np.array(landmarks["right_eye"])], (0, 255, 0))
    nose = landmarks["nose_bridge"] + landmarks["nose_tip"]
    nose_pts = np.array(nose, dtype=np.int32)
    cv2.fillConvexPoly(heatmap, nose_pts, (255, 0, 0))
    mouth = landmarks["top_lip"] + landmarks["bottom_lip"]
    cv2.fillPoly(heatmap, [np.array(mouth)], (0, 0, 255))
    all_points = [pt for pts in landmarks.values() for pt in pts]
    x_vals = [p[0] for p in all_points]
    y_vals = [p[1] for p in all_points]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    hair_roi_top = max(0, min_y - int((max_y - min_y) * 0.8))
    hair_roi_bottom = min_y
    hair_roi_left = max(0, min_x - int((max_x - min_x) * 0.3))
    hair_roi_right = min(w, max_x + int((max_x - min_x) * 0.3))
    hair_roi = rgb_image[hair_roi_top:hair_roi_bottom, hair_roi_left:hair_roi_right]
    if hair_roi.size > 0:
        pixels = hair_roi.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_
        darker_cluster_idx = int(np.argmin(np.sum(centers, axis=1)))
        labels = kmeans.labels_.reshape(hair_roi.shape[:2])
        hair_mask_roi = (labels == darker_cluster_idx).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        hair_mask_roi = cv2.morphologyEx(hair_mask_roi, cv2.MORPH_OPEN, kernel)
        hair_mask_roi = cv2.morphologyEx(hair_mask_roi, cv2.MORPH_CLOSE, kernel)
        full_hair_mask = np.zeros((h, w), dtype=np.uint8)
        full_hair_mask[hair_roi_top:hair_roi_bottom, hair_roi_left:hair_roi_right] = hair_mask_roi
        heatmap[full_hair_mask > 0] = (50, 50, 50)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    heatmap_dir = os.path.join(results_dir, "Feature_Analysis", "Heatmaps")
    overlay_dir = os.path.join(results_dir, "Feature_Analysis", "Overlays")
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(heatmap_dir, f"{base_name}_heatmap.jpg"), heatmap)
    cv2.imwrite(os.path.join(overlay_dir, f"{base_name}_analysis.jpg"), overlay)
    if plt is not None:
        plt.close("all")


def _apply_affine_transform(src, src_tri, dst_tri, size):
    src_tri = np.array(src_tri, dtype=np.float32)
    dst_tri = np.array(dst_tri, dtype=np.float32)
    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)  # type: ignore[arg-type]
    dst = cv2.warpAffine(
        src,
        warp_mat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return dst


def _warp_triangle(img1, img2, t1, t2):
    t1_arr = np.array([t1], dtype=np.float32)
    t2_arr = np.array([t2], dtype=np.float32)
    r1 = cv2.boundingRect(t1_arr)  # type: ignore[arg-type]
    r2 = cv2.boundingRect(t2_arr)  # type: ignore[arg-type]
    t1_rect = []
    t2_rect = []
    t1_rect_int = []
    t2_rect_int = []
    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t1_rect_int.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    t2_rect_int = np.array(t2_rect_int, dtype=np.int32)
    cv2.fillConvexPoly(mask, t2_rect_int, (1.0, 1.0, 1.0), 16, 0)  # type: ignore[arg-type]
    img1_rect = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    size = (r2[2], r2[3])
    img2_rect = _apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask
    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = (
        img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
    )
    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = (
        img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] + img2_rect
    )


def dbscan_smooth_contour(points, eps=5, min_samples=3):
    dense_points = []
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        for t in np.linspace(0, 1, 10):
            x = int(p1[0] * (1 - t) + p2[0] * t)
            y = int(p1[1] * (1 - t) + p2[1] * t)
            dense_points.append([x, y])
            dense_points.append([x + np.random.randint(-2, 3), y + np.random.randint(-2, 3)])
    dense_points = np.array(dense_points)
    db = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(dense_points)
    labels = db.labels_
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    if not unique_labels:
        return points
    largest_cluster_label = max(unique_labels, key=lambda l: np.sum(labels == l))
    cluster_points = dense_points[labels == largest_cluster_label]
    smoothed_points = []
    for p in points:
        dists = np.linalg.norm(cluster_points - p, axis=1)
        closest_idx = int(np.argmin(dists))
        smoothed_points.append(cluster_points[closest_idx])
    return np.array(smoothed_points)


def face_morph_one_image(image_path, output_dir, configs=None):
    _require_face_recognition()
    fr = cast(object, face_recognition)
    img = cv2.imread(image_path)
    if img is None:
        return
    img_orig = img.copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lm_list = cast("object", fr).face_landmarks(rgb)  # type: ignore[attr-defined]
    if not lm_list:
        return
    landmarks = lm_list[0]
    chin = landmarks.get("chin")
    nose_bridge = landmarks.get("nose_bridge")
    if not chin or not nose_bridge:
        return
    chin_points = np.array(chin, dtype=np.int32)
    center_x = float(np.mean([p[0] for p in nose_bridge]))
    center_y = float(np.mean([p[1] for p in nose_bridge]))
    center = np.array([center_x, center_y])
    slim_points = []
    for p in chin_points:
        vec = p - center
        slim_points.append(center + vec * 0.85)
    slim_points = np.array(slim_points, dtype=np.int32)
    fat_points = []
    for p in chin_points:
        vec = p - center
        fat_points.append(center + vec * 1.15)
    fat_points = np.array(fat_points, dtype=np.int32)
    if configs is None:
        configs = [("slim", 5, 3), ("fat", 5, 3), ("slim", 10, 5), ("fat", 10, 5)]
    h, w = img.shape[:2]
    boundary_points = [
        (0, 0),
        (w // 2, 0),
        (w - 1, 0),
        (0, h // 2),
        (w - 1, h // 2),
        (0, h - 1),
        (w // 2, h - 1),
        (w - 1, h - 1),
    ]
    for mode, eps, min_samples in configs:
        target_base = slim_points if mode == "slim" else fat_points
        smoothed_target = dbscan_smooth_contour(target_base, eps=eps, min_samples=min_samples)
        src_points_list = []
        dst_points_list = []
        for i, p in enumerate(chin):
            src_points_list.append(p)
            dst_points_list.append(smoothed_target[i])
        for feature in landmarks.keys():
            if feature == "chin":
                continue
            for p in landmarks[feature]:
                src_points_list.append(p)
                dst_points_list.append(p)
        for p in boundary_points:
            src_points_list.append(p)
            dst_points_list.append(p)
        src_points_arr = np.array(src_points_list, dtype=np.int32)
        dst_points_arr = np.array(dst_points_list, dtype=np.int32)
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)
        for p in dst_points_arr:
            px = int(np.clip(p[0], 0, w - 1))
            py = int(np.clip(p[1], 0, h - 1))
            subdiv.insert((float(px), float(py)))
        triangle_list = subdiv.getTriangleList()
        output_img = np.zeros(img.shape, dtype=img.dtype)
        def get_index(pt, points):
            for ii, pp in enumerate(points):
                if abs(pp[0] - pt[0]) < 1 and abs(pp[1] - pt[1]) < 1:
                    return ii
            return -1
        for t in triangle_list:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            idx1 = get_index(pt1, dst_points_arr)
            idx2 = get_index(pt2, dst_points_arr)
            idx3 = get_index(pt3, dst_points_arr)
            if idx1 == -1 or idx2 == -1 or idx3 == -1:
                continue
            t1 = [src_points_arr[idx1], src_points_arr[idx2], src_points_arr[idx3]]
            t2 = [dst_points_arr[idx1], dst_points_arr[idx2], dst_points_arr[idx3]]
            _warp_triangle(img_orig, output_img, t1, t2)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        save_name = f"{name}_{mode}_feps{eps}_fmin{min_samples}{ext}"
        cv2.imwrite(os.path.join(output_dir, save_name), output_img)


def run_face_feature_analysis_on_dir(input_dir, results_dir, recursive=False):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    if recursive:
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    else:
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths = [p for p in image_paths if "montage" not in os.path.basename(p).lower()]
    for p in tqdm(image_paths, desc="Analyzing Features", unit="img"):
        analyze_face_features(p, results_dir)


def run_face_morphing_on_dir(input_dir, output_dir, recursive=False, mode="both"):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    if recursive:
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    else:
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths = [p for p in image_paths if "montage" not in os.path.basename(p).lower()]
    if mode == "slim":
        configs = [("slim", 5, 3), ("slim", 10, 5)]
    elif mode == "fat":
        configs = [("fat", 5, 3), ("fat", 10, 5)]
    else:
        configs = None
    for p in tqdm(image_paths, desc="Face Morphing", unit="img"):
        rel_dir = os.path.relpath(os.path.dirname(p), input_dir)
        out_dir = os.path.join(output_dir, rel_dir)
        face_morph_one_image(p, out_dir, configs=configs)
