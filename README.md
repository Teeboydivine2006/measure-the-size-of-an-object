# Measure the Size of an Object

Measure the real-world dimensions (in centimeters) of an object in a photo, using computer vision — no tape measure required.

The pipeline combines three techniques:
- **YOLO11** — detects the object in the image and draws a bounding box around it.
- **SAM 2** (Segment Anything Model 2) — turns that rough box into a precise, pixel-accurate mask of the object's actual shape.
- **ArUco marker calibration** — a printed reference marker of known size (5cm x 5cm) in the same photo lets the pipeline convert pixel measurements into real-world centimeters, regardless of camera distance or angle.

## How it works

```
 Input photo (object + ArUco marker)
            │
            ▼
 ┌─────────────────────┐
 │ 1. Detect object     │  detect_object.py        → bounding box (YOLO11)
 └─────────────────────┘
            │
            ▼
 ┌─────────────────────┐
 │ 2. Segment object    │  segment_object.py        → precise mask (SAM 2)
 └─────────────────────┘
            │
            ▼
 ┌─────────────────────┐
 │ 3. Measure in pixels │  find_dimension_pixels.py  → width/height in px
 └─────────────────────┘
            │
            ▼
 ┌─────────────────────┐
 │ 4. Calibrate         │  pixels_to_real_world.py   → pixels-per-cm (ArUco)
 └─────────────────────┘
            │
            ▼
   Real-world width x height, in cm
```

`measure_size_of_any_object.py` orchestrates all four steps into one pipeline.

## Project structure

```
measure-the-size-of-an-object/
├── model/
│   ├── yolo11n.pt              # YOLO11 nano weights
│   └── sam2_b.pt                # SAM 2.1 base+ checkpoint
├── src/
│   └── measure_the_size_of_an_object/
│       ├── detect_object.py
│       ├── segment_object.py
│       ├── find_dimension_pixels.py
│       ├── pixels_to_real_world.py
│       └── measure_size_of_any_object.py
├── pyproject.toml
└── README.md
```

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A printed ArUco marker (see **Calibration setup** below)

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/measure-the-size-of-an-object.git
cd measure-the-size-of-an-object
uv add opencv-contrib-python numpy ultralytics torch torchvision
uv add "git+https://github.com/facebookresearch/sam2.git"
```

### Download the model weights

The weight files are not committed to this repo (they're large binary files — not a good fit for git). Download them into `model/`:

```bash
mkdir -p model
cd model

# YOLO11 nano — object detection
curl -L -O https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# SAM 2.1 base+ — segmentation (rename to match the path used in segment_object.py)
curl -L -o sam2_b.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

cd ..
```

## Calibration setup

Real-world measurement requires a reference object of known size in every photo: a printed **ArUco marker** (dictionary `DICT_6X6_250`, id `0`), 5cm x 5cm.

1. Print the marker at **100% / "actual size"** — do not let your printer "fit to page" or scale it.
2. Measure the printed square with a ruler to confirm it's genuinely 5cm. Printers aren't always perfectly accurate — if yours comes out a different size, update `marker_size_cm` in `pixels_to_real_world.py` to match.
3. Place the marker flat, in the same plane as the object you're measuring (same surface, similar distance from the camera).
4. Take one photo containing **both** the marker and the object you want measured.

## Usage

> The exact command-line interface below should match however `measure_size_of_any_object.py` is wired up in this repo — update this section once that script's arguments are finalized.

```bash
uv run python src/measure_the_size_of_an_object/measure_size_of_any_object.py --image path/to/photo.jpg
```

This will:
1. Detect the object in the photo.
2. Segment it precisely with SAM 2.
3. Measure its width and height in pixels.
4. Calibrate against the ArUco marker in the same photo.
5. Print (and optionally save/draw) the object's real-world dimensions in centimeters.

### Running individual stages

Each stage can also be run and tested on its own — useful for debugging a specific part of the pipeline:

```bash
uv run python src/measure_the_size_of_an_object/pixels_to_real_world.py
uv run python src/measure_the_size_of_an_object/detect_object.py
uv run python src/measure_the_size_of_an_object/segment_object.py
uv run python src/measure_the_size_of_an_object/find_dimension_pixels.py
```

## Known limitations

- Works best on real photographs. Illustrations, renders, or heavily stylized images may not be reliably detected by YOLO, since it was trained on real-world photos.
- Measurement accuracy depends on how precisely the ArUco marker was printed and how flat/aligned it is relative to the object in frame.
- Objects photographed at extreme angles may still introduce some measurement error, even with marker-based calibration.

## Acknowledgments

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) for object detection.
- [Meta AI's SAM 2](https://github.com/facebookresearch/sam2) for segmentation.
- OpenCV's ArUco module for marker-based calibration.
