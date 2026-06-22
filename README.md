
# A VIDEO-TO-SIMULATION FRAMEWORK FOR SOCIAL FORCE PEDESTRIAN MODEL PARAMETER CALIBRATION


This repository provides a workflow to generate a pedestrian trajectory dataset from a video and calibrate a Social Force Model (SFM) using the generated data.

## Repository Structure

```text
.
├── generate_dataset_from_video.py
├── calibrate_sfm.py
├── example.mp4
└── README.md
```

## Requirements

- Python 3.10.19
- Virtual environment (recommended)

## Step 1: Create Environment and Install Dependencies

### Using Conda

```bash
conda create -n myenv python=3.10.19
conda activate myenv
```

### Using Python venv (Windows)

```bash
python -m venv myenv
myenv\Scripts\activate
```

### Using Python venv (macOS/Linux)

```bash
python3 -m venv myenv
source myenv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Step 2: Preprocess Video

Place `example.mp4` in the repository root directory.

Open `generate_dataset_from_video.py` and set:

```python
VIDEO_IN = "example.mp4"
```

Configure the following parameters:

- Region of interest (ROI) bounding box
- Source points
- Destination points

Run the script:

```bash
python generate_dataset_from_video.py
```

The script generates:

- `sample_frame.jpg` – ROI preview on the original frame
- `transformed_sample_frame.jpg` – Perspective-transformed preview

Adjust the ROI and transformation coordinates until the transformed view correctly aligns with the pedestrian walking area.

## Step 3: Generate SFM Training Dataset

Once the ROI and transformation settings are satisfactory, enable dataset generation in `generate_dataset_from_video.py`:

```python
GENERATE_DATA = True
```

Run the script again:

```bash
python generate_dataset_from_video.py
```

This generates:

- `pedestrian_dataset.json`
- Perspective-transformed video output
- Additional intermediate files (if enabled)

The generated dataset contains:

- Detected pedestrian trajectories
- Pedestrian coordinates
- Homography transformation matrix

## Step 4: Calibrate the Social Force Model

Open `calibrate_sfm.py` and set:

```python
DATASET_JSON = "pedestrian_dataset.json"
```

Run the calibration script:

```bash
python calibrate_sfm.py
```

During training, the script displays:

- Current epoch
- Training loss
- Updated Social Force Model parameters

After calibration is completed, record the best-performing parameter values for future simulations and analysis.

## Workflow Summary

1. Create and activate a Python 3.10.19 environment.
2. Place `example.mp4` in the repository root directory.
3. Configure the ROI, source points, and destination points in `generate_dataset_from_video.py`.
4. Verify the transformation using the generated preview images.
5. Set `GENERATE_DATA = True` and generate the dataset.
6. Run `calibrate_sfm.py` using the generated `pedestrian_dataset.json`.
7. Save the best calibrated SFM parameters from the training output.
