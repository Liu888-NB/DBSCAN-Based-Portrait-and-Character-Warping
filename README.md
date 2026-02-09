# DBSCAN-Based Portrait and Character Warping

This repository contains our Data Mining course project, which explores how **DBSCAN clustering** can be used as a **structure-aware deformation controller** for both **portrait images** and **character images**.

By combining clustering, pose estimation, and image processing techniques, we achieve controllable and natural-looking deformation effects such as body slimming/fattening and character stroke thickness adjustment.

---

## Project Overview

The project focuses on three main tasks:

- **Human Detection**
- **Portrait Distortion**
- **Character Weight Adjustment**

Instead of using DBSCAN as a pure segmentation tool, we reinterpret its parameters as a way to **control deformation strength**, bridging data mining methods with pixel-level image transformation.

---

## Human Detection

### Motivation

Directly applying DBSCAN to all image pixels fails due to:
- Background interference
- Shadows and texture noise
- Inaccurate human-body separation

### Solution: MediaPipe + Targeted Clustering

- **MediaPipe Pose** is used to extract **33 human body keypoints** (head, torso, limbs, joints).
- **Dlib** is used for facial feature detection.
- DBSCAN is applied **only to these keypoints**, not raw pixels.

This approach:
- Eliminates background noise
- Restricts deformation to the human body
- Enables structure-aware deformation control

---

## DBSCAN’s Role Redefined

In this project, DBSCAN is **not used to find objects**, but to:

- Identify the **main body structure** (e.g., trunk cluster)
- Assign **different deformation weights** to different regions
- Convert clustering parameters into deformation strength

**Parameter Mapping:**
- Larger `eps` / `MinPts` → stronger deformation
- Smaller `eps` / `MinPts` → subtle, fine-grained adjustment

---

## Portrait Distortion

Portrait deformation is implemented using **Liquify Warp** combined with DBSCAN-controlled parameters.

### Warp Modes

- **Local Radial Warp (Limbs)**  
  Natural slimming/fattening around joints

- **Segment-Based Warp Along Bones**  
  Continuous thickness adjustment along limbs

- **Global Torso Warp**  
  Smooth chest, waist, and belly deformation

- **Head & Face Warp**  
  Face width and head size adjustment without vertical distortion

### Parameter Analysis

A parameter grid (`eps × MinPts`) is used to visualize deformation results and analyze sensitivity.

**Recommended Settings:**

| Target Effect | eps | MinPts |
|--------------|-----|--------|
| Natural fine-tune | 30–50 | 3–5 |
| Balanced | 50–70 | 5–8 |
| Intense slimming/fattening | 70–90 | 8–12 |

---

## Character Weight Adjustment

For character images, the pipeline is simplified due to their structural properties.

### Key Characteristics

- Dark strokes on light background
- No need for pose or facial detection
- Pixel intensity is sufficient for segmentation

### Processing Pipeline

1. Foreground extraction (stroke pixels)
2. DBSCAN clustering on foreground pixels
3. Parameter-based strength mapping
4. Morphological operations:
   - **Erosion** for slim mode
   - **Dilation** for fat mode

DBSCAN clusters allow **local stroke-level control**, preserving overall character integrity.

---

## Results

- DBSCAN parameters provide smooth and interpretable control over deformation intensity
- Portrait deformation remains anatomically consistent
- Character stroke adjustment preserves local geometry
- One unified clustering-based framework supports multiple deformation tasks

---

## Conclusion

This project demonstrates how **DBSCAN clustering** can be repurposed from traditional segmentation into a **deformation strength controller**. By integrating MediaPipe pose estimation for portraits and morphology-based processing for characters, we build a flexible and effective image deformation system that combines data mining principles with computer vision techniques.

---

## Authors

Group 32

- Jiaming Lei  
- Yichang Qiao  
- Chenrui Liu
