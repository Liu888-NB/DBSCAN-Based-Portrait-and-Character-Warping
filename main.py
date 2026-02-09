import argparse
import glob
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from body import BodyReshaper, build_param_grid, run_body_analysis_on_dir, run_body_reshape_best_on_dir
from shared import (
    cluster_faces,
    encode_faces,
    run_face_feature_analysis_on_dir,
    run_face_morphing_on_dir,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=[
            "reshape_grid",
            "reshape_best",
            "encode_faces",
            "cluster_faces",
            "feature_analysis",
            "body_analysis",
            "face_morphing",
            "pipeline",
            "dataset_pipeline",
        ],
        default="reshape_grid",
        help="which task to run (default: reshape_grid)",
    )
    parser.add_argument("--mode", choices=["slim", "fat", "both"], default="both",
                        help="which mode to export: slim / fat / both")
    parser.add_argument("--input_dir", default=None,
                        help="input folder. If omitted, prefer Results/Clusters, else Photos/dataset")
    parser.add_argument("--output_dir", default=None,
                        help="output folder (default: Results/Body_Reshaping_Grids)")
    parser.add_argument("--recursive", action="store_true",
                        help="scan input_dir recursively (useful for Results/Clusters/Cluster_*)")
    parser.add_argument("--photos_dir", default=None, help="Photos/dataset folder for encode_faces")
    parser.add_argument("--encodings_path", default=None, help="Results/encodings.pickle path")
    parser.add_argument("--detection_method", choices=["hog", "cnn"], default="hog")
    parser.add_argument("--max_detect_size", type=int, default=800,
                        help="max image side for cnn face detection to reduce GPU OOM (default: 800)")
    parser.add_argument("--cluster_eps", type=float, default=0.5)
    parser.add_argument("--cluster_min_samples", type=int, default=1)
    parser.add_argument("--results_dir", default=None, help="Results folder (default: Results)")
    parser.add_argument("--slim_eps", type=float, default=60.0, help="best slim eps (recommended 50-70)")
    parser.add_argument("--slim_minpts", type=int, default=6, help="best slim MinPts (recommended 5-8)")
    parser.add_argument("--fat_eps", type=float, default=40.0, help="best fat eps (recommended 30-50)")
    parser.add_argument("--fat_minpts", type=int, default=4, help="best fat MinPts (recommended 3-5)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_clusters_root = os.path.join(project_root, "Results", "Clusters")
    default_photos_root = os.path.join(project_root, "Photos", "dataset")
    default_results_dir = os.path.join(project_root, "Results")
    default_encodings_path = os.path.join(default_results_dir, "encodings.pickle")

    photos_dir = args.photos_dir or default_photos_root
    results_dir = args.results_dir or default_results_dir
    encodings_path = args.encodings_path or default_encodings_path

    if args.task == "encode_faces":
        encode_faces(
            photos_dir,
            encodings_path,
            detection_method=args.detection_method,
            max_detect_size=args.max_detect_size,
        )
        return

    if args.task == "cluster_faces":
        os.makedirs(results_dir, exist_ok=True)
        cluster_faces(
            encodings_path,
            results_dir,
            eps=args.cluster_eps,
            min_samples=args.cluster_min_samples,
        )
        return

    if args.task == "feature_analysis":
        os.makedirs(results_dir, exist_ok=True)
        input_dir = args.input_dir or default_clusters_root
        run_face_feature_analysis_on_dir(input_dir, results_dir, recursive=args.recursive)
        return

    if args.task == "body_analysis":
        input_dir = args.input_dir or default_photos_root
        out_dir = args.output_dir or os.path.join(results_dir, "Body_Analysis")
        os.makedirs(out_dir, exist_ok=True)
        run_body_analysis_on_dir(input_dir, out_dir, recursive=args.recursive)
        print(f"[INFO] Done. Body analysis saved to: {out_dir}")
        return

    if args.task == "face_morphing":
        input_dir = args.input_dir or default_clusters_root
        out_dir = args.output_dir or os.path.join(results_dir, "Morphing")
        os.makedirs(out_dir, exist_ok=True)
        run_face_morphing_on_dir(input_dir, out_dir, recursive=args.recursive, mode=args.mode)
        return

    if args.task == "pipeline":
        os.makedirs(results_dir, exist_ok=True)
        encode_faces(
            photos_dir,
            encodings_path,
            detection_method=args.detection_method,
            max_detect_size=args.max_detect_size,
        )
        cluster_faces(encodings_path, results_dir, eps=args.cluster_eps, min_samples=args.cluster_min_samples)
        clusters_root = os.path.join(results_dir, "Clusters")
        run_face_feature_analysis_on_dir(clusters_root, results_dir, recursive=True)
        morph_root = os.path.join(results_dir, "Morphing")
        os.makedirs(morph_root, exist_ok=True)
        run_face_morphing_on_dir(clusters_root, morph_root, recursive=True, mode="both")
        args.input_dir = morph_root
        args.output_dir = os.path.join(results_dir, "Body_Reshaping_Grids")
        args.recursive = True
        args.task = "reshape_grid"

    if args.task == "dataset_pipeline":
        input_dir = args.input_dir or default_photos_root
        os.makedirs(results_dir, exist_ok=True)
        run_face_feature_analysis_on_dir(input_dir, results_dir, recursive=args.recursive)
        body_out = os.path.join(results_dir, "Body_Analysis")
        os.makedirs(body_out, exist_ok=True)
        run_body_analysis_on_dir(input_dir, body_out, recursive=args.recursive)
        best_out = args.output_dir or os.path.join(results_dir, "Body_Reshaping_Best")
        os.makedirs(best_out, exist_ok=True)
        run_body_reshape_best_on_dir(
            input_dir=input_dir,
            output_dir=best_out,
            recursive=args.recursive,
            slim_eps=args.slim_eps,
            slim_minpts=args.slim_minpts,
            fat_eps=args.fat_eps,
            fat_minpts=args.fat_minpts,
            export_mode=args.mode,
        )
        print(f"[INFO] Done. Face analysis in: {os.path.join(results_dir, 'Feature_Analysis')}")
        print(f"[INFO] Done. Body analysis in: {body_out}")
        print(f"[INFO] Done. Best reshape in: {best_out}")
        return

    if args.task == "reshape_best":
        photos_root = args.input_dir if args.input_dir is not None else (
            default_clusters_root if os.path.isdir(default_clusters_root) else default_photos_root
        )
        output_dir = args.output_dir or os.path.join(results_dir, "Body_Reshaping_Best")
        os.makedirs(output_dir, exist_ok=True)
        run_body_reshape_best_on_dir(
            input_dir=photos_root,
            output_dir=output_dir,
            recursive=args.recursive,
            slim_eps=args.slim_eps,
            slim_minpts=args.slim_minpts,
            fat_eps=args.fat_eps,
            fat_minpts=args.fat_minpts,
            export_mode=args.mode,
        )
        print(f"[INFO] Done. Best results saved to: {output_dir}")
        return

    photos_root = args.input_dir if args.input_dir is not None else (
        default_clusters_root if os.path.isdir(default_clusters_root) else default_photos_root
    )
    output_dir = args.output_dir or os.path.join(project_root, "Results", "Body_Reshaping_Grids")
    os.makedirs(output_dir, exist_ok=True)

    reshaper = BodyReshaper()

    print(f"[INFO] Scanning folder: {photos_root}")
    exts = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    if args.recursive:
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(photos_root, "**", ext), recursive=True))
    else:
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(photos_root, ext)))

    print(f"[INFO] Found {len(image_paths)} images")
    if len(image_paths) == 0:
        print("[WARNING] No images found! Check input_dir and extensions.")
        return
    eps_list = [5, 10, 15, 25]
    minpts_list = [2, 4, 6, 8]
    print(f"[INFO] Using eps list: {eps_list}")
    print(f"[INFO] Using MinPts list: {minpts_list}")

    import cv2
    from tqdm import tqdm

    for img_path in tqdm(image_paths, desc="Processing Images", unit="img"):
        image = cv2.imread(img_path)
        if image is None:
            continue
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        if args.mode in ["slim", "both"]:
            slim_grid = build_param_grid(image, reshaper, eps_list, minpts_list, mode="slim")
            cv2.imwrite(os.path.join(output_dir, f"{name}_slim_grid{ext}"), slim_grid)
        if args.mode in ["fat", "both"]:
            fat_grid = build_param_grid(image, reshaper, eps_list, minpts_list, mode="fat")
            cv2.imwrite(os.path.join(output_dir, f"{name}_fat_grid{ext}"), fat_grid)
    print(f"[INFO] Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
