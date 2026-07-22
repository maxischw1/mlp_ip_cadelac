# CaDeLaC

Source code for **Context-Aware Deep Lagrangian Networks for Model Predictive Control (CaDeLaC)**.

This fork extends the original CaDeLaC repository with a simplified **2-DOF hip-knee exoskeleton setup** and an experimental **Context-Aware LSTM + Direct Torque MLP** architecture.

The original CaDeLaC implementation learns physics-consistent dynamics through a Deep Lagrangian Network. The direct-torque extension implemented in this fork instead uses:

- an LSTM to infer a latent context vector \(z\) from motion history,
- a feedforward MLP to map the current motion state and context directly to joint torque,
- subject-wise evaluation on an unseen exoskeleton participant.

> [!NOTE]
> The original DeLaN implementation remains available in the repository.  
> The currently active experimental path uses the direct torque MLP.

---

## Citation

When using the original CaDeLaC method, please cite:

```bibtex
@inproceedings{schulze2025contextawaredelan,
  author={Schulze, Lucas and Peters, Jan and Arenz, Oleg},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title={Context-Aware Deep Lagrangian Networks for Model Predictive Control},
  year={2025},
  pages={6939--6946},
  doi={10.1109/IROS60139.2025.11246292}
}
```

Experiment videos for the original project are available on the
[CaDeLaC project website](https://schulze18.github.io/cadelac_website/).

---

## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Original CaDeLaC Usage](#original-cadelac-usage)
- [Exoskeleton Extension](#exoskeleton-extension)
  - [Research Goal](#research-goal)
  - [Architecture](#architecture)
  - [Current Configuration](#current-configuration)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Recorded Results](#recorded-results)
  - [Interpretation and Limitations](#interpretation-and-limitations)
- [Repository Structure](#repository-structure)
- [Repository Hygiene](#repository-hygiene)
- [License](#license)

---

# Installation

## Training and Simulation Setup

### 1. Clone the Repository

```bash
git clone git@github.com:maxischw1/mlp_ip_cadelac.git
cd mlp_ip_cadelac
git submodule update --recursive --init
```

### 2. Create the Conda Environment

```bash
conda env create -f cadelac_env.yml
conda activate cadelac
```

### 3. Install CaDeLaC and Additional Dependencies

```bash
pip install -e .
pip install l4casadi==1.4.1 --no-build-isolation
```

### 4. Install Acados

The repository currently uses Acados `v0.4.3`.

Follow the
[official Acados installation guide](https://docs.acados.org/installation/).

```bash
cd acados
mkdir -p build
cd build

cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
```

Install the Python interface:

```bash
cd ../../
pip install -e acados/interfaces/acados_template
```

Add the following environment variables to `~/.bashrc`:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"
```

Reload the shell configuration:

```bash
source ~/.bashrc
```

---

## Real Robot Setup

### 1. Create a ROS Workspace

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

git clone git@github.com:maxischw1/mlp_ip_cadelac.git
cd mlp_ip_cadelac

git submodule update --recursive --init
```

### 2. Create the ROS Conda Environment

```bash
conda env create -f cadelac_ros_env.yml
conda activate cadelac_ros
```

### 3. Configure the Compiler

To ensure that Acados and Libfranka use the same compiler:

```bash
export CC="$HOME/miniconda3/envs/cadelac_ros/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$HOME/miniconda3/envs/cadelac_ros/bin/x86_64-conda-linux-gnu-g++"
```

These variables may also be added to `~/.bashrc`.

### 4. Install CaDeLaC

```bash
pip install -e .
pip install l4casadi==1.4.1 --no-build-isolation
```

### 5. Install Libfranka

```bash
git clone --recurse-submodules https://github.com/frankarobotics/libfranka.git
cd libfranka

git checkout 0.13.3
git submodule update

mkdir -p build
cd build

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/openrobots/lib/cmake \
  -DBUILD_TESTS=OFF \
  ..

make
```

### 6. Build the ROS Workspace

```bash
cd ~/catkin_ws

catkin_make \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFranka_DIR:PATH="<path-to-libfranka>/libfranka/build" \
  -j2

source devel/setup.bash
```

---

# Datasets

The datasets used by the original CaDeLaC experiments are available from
[Hugging Face](https://huggingface.co/datasets/schulze18/cadelac).

```bash
git clone \
  https://huggingface.co/datasets/schulze18/cadelac \
  cadelac/learning/datasets/
```

The exoskeleton extension uses separately generated `.pkl` datasets described in
[Dataset Preparation](#dataset-preparation).

---

# Original CaDeLaC Usage

## Evaluate the Pretrained IROS 2025 Model

```bash
python -m cadelac.learning.train_panda -l 2
```

## Train Residual Dynamics

```bash
python -m cadelac.learning.train_panda -l 0
```

## Evaluate a Trained Residual Model

```bash
python -m cadelac.learning.train_panda -l 1
```

## Train a Full DeLaN Robot Model

```bash
python -m cadelac.learning.train_panda -l 0 -f 1
```

## Evaluate a Full DeLaN Robot Model

```bash
python -m cadelac.learning.train_panda -l 1 -f 1
```

> [!NOTE]
> The dataset can require several minutes to load.  
> The original paper trained for 3000 epochs, although shorter runs may already
> provide useful performance.

---

## MuJoCo Controller Evaluation

Evaluate joint tracking under random loads:

```bash
python -m cadelac.control.eval_controllers_multiple_envs -c 2
```

Controller options:

| Argument | Controller |
|---:|---|
| `-c 0` | Nominal MPC |
| `-c 1` | MPC + EKF |
| `-c 2` | CaDeLaC context-aware MPC |

---

## Real Franka Robot

### Terminal 1: Launch the Hardware Interface

```bash
conda activate cadelac_ros
source ~/catkin_ws/devel/setup.bash

roslaunch \
  franka_example_controllers \
  effort_joint_controller.launch \
  robot_ip:="<robot-ip>" \
  load_gripper:=true \
  robot:=panda
```

### Terminal 2: Run the CaDeLaC Controller

```bash
conda activate cadelac_ros
source ~/catkin_ws/devel/setup.bash

cd ~/catkin_ws/src/mlp_ip_cadelac

python -m cadelac.ros.cadelac_node -c 2
```

> [!WARNING]
> Acados compiles a controller during its first execution. The initial compilation
> may require several minutes. Follow the safety procedure implemented in
> `cadelac/ros/cadelac_node.py` before executing the controller on hardware.

---

# Exoskeleton Extension

This fork extends the original CaDeLaC repository with a simplified 2-DOF
hip-knee exoskeleton experiment.

The currently active experimental architecture combines:

- an LSTM that estimates a latent context vector from motion history,
- a direct torque MLP with four hidden layers of 64 neurons,
- subject-wise evaluation on the unseen participant BT24.

The currently configured direct torque network is:

```text
16 -> 64 -> 64 -> 64 -> 64 -> 2
```

The two outputs represent the predicted left hip and left knee torque.

> [!IMPORTANT]
> The current direct torque MLP is a black-box baseline and does not use the
> physics-consistent Hessian-based DeLaN prediction path.

---

## Current Architecture

The model implementation is located in:

```text
cadelac/learning/models/context_aware_delan.py
```

The current TensorBoard-enabled training configuration is located in:

```text
cadelac/learning/train_panda_v2.py
```

For every sample, the model uses:

```text
History:
15 previous timesteps of q and qdot

Current state:
q, qdot and qddot

Target:
measured hip and knee torque
```

The data flow is:

```text
15 x [q_hip, q_knee, qdot_hip, qdot_knee]
                         |
                         v
                       LSTM
                         |
                         v
                latent context z
                    10 values
                         |
                         +-------------------------+
                                                   |
Current state:                                     |
[q_hip, q_knee,                                    |
 qdot_hip, qdot_knee,                              |
 qddot_hip, qddot_knee]                            |
         |                                         |
         +-----------------------------------------+
                         |
                         v
        [q, qdot, qddot, z] = 16 values
                         |
                         v
          16 -> 64 -> 64 -> 64 -> 64 -> 2
                         |
                         v
       predicted hip and knee torque
```

### Current Model Settings

| Parameter | Current value |
|---|---:|
| Degrees of freedom | `2` |
| Modeled joints | left hip and left knee |
| History length | `15` timesteps |
| LSTM history features | `q` and `qdot` |
| LSTM input dimension | `4` per timestep |
| LSTM hidden dimension | `10` |
| LSTM depth | `5` layers |
| Context dimension | `10` |
| Current-state dimension | `6` |
| Complete MLP input dimension | `16` |
| MLP architecture | `[64, 64, 64, 64]` |
| MLP output dimension | `2` |
| Hidden activation | `Tanh` |
| Artificial data noise | disabled |
| Power loss | disabled |
| Held-out subject | `BT24` |

---

## Current Assumptions and Limitations

The current experimental setup makes the following assumptions:

- only the left hip and left knee are modeled,
- the measured joint moments are used as torque targets,
- the current torque is not used as a neural-network input,
- historical torque values are not passed to the LSTM,
- only historical `q` and `qdot` values are used to estimate the context,
- the current `q`, `qdot` and `qddot` values are passed directly to the MLP,
- all BT24 samples are excluded from training,
- BT24 is used only for subject-wise evaluation,
- artificial noise is disabled because the dataset already contains real
  measurement noise,
- the direct MLP receives raw state values,
- the sine/cosine input transformation is disabled for the direct MLP,
- the torque loss is normalized using the torque variance of the training set.

The existing data loader still creates and returns torque histories because its
interface is shared with the original CaDeLaC pipeline. These torque histories
are currently not passed to the LSTM.

For the simplified residual setup:

```python
diff_tau = tau
```

The residual target used by the existing training pipeline is therefore equal
to the measured torque.

---

## Legacy DeLaN Components

The class `ContextAwareDeLaN` still constructs the original:

- inertia network,
- potential-energy network,
- direct torque network.

The active model path is selected through:

```python
self.dyn_model = self.dyn_model_mlp
```

Therefore, the current torque prediction uses only:

```text
LSTM context encoder
+
direct torque MLP
```

The inertia and potential-energy networks are still instantiated for
compatibility with the original implementation.

As a result, these inactive components may still:

- appear in the total parameter count,
- appear in the model `state_dict`,
- be stored in checkpoints and final model files.

They are not used to calculate the active `tau_pred` output and do not receive
gradients through the direct-torque prediction path.

The current MLP also does not enforce a physical decomposition into:

```text
inertial torque
Coriolis and centrifugal torque
gravitational torque
```

Legacy evaluation code may still calculate apparent component values by setting
selected inputs to zero. These values are not guaranteed to represent a
physically valid DeLaN decomposition.

The primary valid output of the current baseline is therefore:

```text
predicted total hip and knee torque
```

---

# Dataset Preparation

Dataset-generation scripts are located in:

```text
scripts/data/
```

Generated datasets are written to:

```text
cadelac/learning/datasets/panda/
```

## `scripts/data/make_exo_pkl.py`

Creates a basic single-recording 2-DOF exoskeleton dataset from matching:

```text
Exo.csv
Joint_Moments_Filt.csv
```

The script:

- reads left hip and knee angles,
- reads or derives joint velocities,
- calculates joint accelerations,
- reads filtered hip and knee moments,
- converts angular values from degrees to radians,
- writes the data in the format expected by the training pipeline.

Run from the repository root:

```bash
python scripts/data/make_exo_pkl.py
```

---

## `scripts/data/make_all_exo_pkls.py`

Searches the configured source directory recursively for matching:

```text
Exo.csv
Joint_Moments_Filt.csv
```

The script:

- creates left-leg and right-leg datasets,
- segments longer recordings into shorter trials,
- creates the Context-Aware dataset,
- sets `diff_tau` equal to the measured torque.

Run:

```bash
python scripts/data/make_all_exo_pkls.py
```

The current training configuration uses:

```text
cadelac/learning/datasets/panda/
exo_hip_knee_delan_2dof_left_all_trials_context.pkl
```

---

## `scripts/data/fix_exo_pkl_time.py`

Repairs inconsistent time information in an existing generated dataset.

The script can:

- replace an invalid or nonuniform time axis,
- enforce the intended sampling interval,
- recompute joint accelerations from joint velocities,
- create a backup of the original dataset.

Run:

```bash
python scripts/data/fix_exo_pkl_time.py
```

> [!CAUTION]
> Use this script only when the generated dataset contains inconsistent or
> incorrect timestamp information.

---

# Training

## `cadelac/learning/train_panda.py`

This is the earlier training entry point retained for compatibility with the
existing CaDeLaC workflow.

It uses the same `ContextAwareDeLaN` model class, but its hyperparameters and
training duration are configured separately from `train_panda_v2.py`.

---

## `cadelac/learning/train_panda_v2.py`

This is the current TensorBoard-enabled training script.

It performs the following steps:

1. loads the Context-Aware exoskeleton dataset,
2. identifies all labels containing `BT24`,
3. excludes BT24 from the training data,
4. creates 15-timestep `q` and `qdot` history windows,
5. creates the current `q`, `qdot`, `qddot` state,
6. trains the LSTM and direct torque MLP,
7. evaluates the current model on BT24,
8. logs metrics to TensorBoard,
9. saves checkpoints and the final model.

Start a new training run from the repository root:

```bash
mkdir -p logs runs

python -u -m cadelac.learning.train_panda_v2 \
  -c 1 \
  -i 0 \
  -s 0 \
  -r 0 \
  -l 0 \
  -m 1 \
  -f 0 \
  --tb-eval-period 10 \
  --tb-histogram-period 1000 \
  2>&1 | tee logs/train_panda_v2.log
```

### Training Arguments

| Argument | Meaning |
|---:|---|
| `-c 1` | use CUDA when available |
| `-i 0` | use CUDA device `0` |
| `-s 0` | use random seed `0` |
| `-r 0` | disable interactive plot rendering |
| `-l 0` | start a new model |
| `-m 1` | save checkpoints and the final model |
| `-f 0` | use the Context-Aware residual branch |
| `--tb-eval-period 10` | evaluate BT24 every 10 epochs |
| `--tb-histogram-period 1000` | log parameter histograms periodically |

Models and checkpoints are stored below:

```text
cadelac/learning/trained_models/res_model/panda/ContextAware/
```

---

# TensorBoard

TensorBoard event files are written to:

```text
runs/
```

Start TensorBoard from the repository root:

```bash
tensorboard --logdir runs --port 6006
```

Then open:

```text
http://localhost:6006
```

The main training and evaluation metrics are:

```text
Loss/train_torque_normalized
Loss/BT24_torque_normalized
```

Additional metrics include:

```text
Train/normalized_hip_mse
Train/normalized_knee_mse

BT24/normalized_hip_mse
BT24/normalized_knee_mse

BT24/raw_hip_mse
BT24/raw_knee_mse

Context/BT24_z_mean
Context/BT24_z_std

Timing/seconds_per_epoch
Optimization/learning_rate
```

The training loss is written after every epoch.

The BT24 loss is written according to:

```text
--tb-eval-period
```

When multiple TensorBoard runs are selected, TensorBoard displays one training
and one BT24 curve for every selected model.

---

# Evaluation

Evaluation scripts are located in:

```text
scripts/evaluation/
```

Set the model that should be evaluated:

```bash
MODEL_PATH="<path-to-model.torch>"
```

## Checkpoint Evaluation

Evaluate available Context-Aware checkpoints:

```bash
python scripts/evaluation/evaluate_context_checkpoints.py
```

Plot the generated checkpoint metrics:

```bash
python scripts/evaluation/plot_context_checkpoint_metrics.py
```

---

## Torque Prediction Plot

Create a measured-versus-predicted torque plot:

```bash
python scripts/evaluation/plot_zoomed_context_torque_prediction.py \
  --model "$MODEL_PATH" \
  --output-dir <output-directory>
```

Evaluate only Ball Toss:

```bash
python scripts/evaluation/plot_zoomed_context_torque_prediction.py \
  --model "$MODEL_PATH" \
  --segment ball_toss \
  --output-dir <output-directory>
```

Evaluate only Incline Walk:

```bash
python scripts/evaluation/plot_zoomed_context_torque_prediction.py \
  --model "$MODEL_PATH" \
  --segment incline_walk \
  --output-dir <output-directory>
```

---

## Left-Leg Torque Grid

Create a Ball-Toss torque grid:

```bash
python scripts/evaluation/plot_left_leg_torque_grid.py \
  --model "$MODEL_PATH" \
  --movement ball_toss \
  --output-dir <output-directory>
```

Create an Incline-Walk torque grid:

```bash
python scripts/evaluation/plot_left_leg_torque_grid.py \
  --model "$MODEL_PATH" \
  --movement incline_walk \
  --output-dir <output-directory>
```

---

# Interpreting the Results

The main model comparison should use:

- normalized training torque loss,
- normalized BT24 torque loss,
- separate hip and knee errors,
- measured-versus-predicted torque plots,
- movement-specific evaluation plots.

A decreasing training loss indicates that the model is fitting the training
participants.

A decreasing BT24 loss indicates improved generalization to the unseen BT24
participant.

When the training loss decreases while the BT24 loss remains constant or
increases, the model may be overfitting to the training participants.

Individual completed training runs and their numerical results are
intentionally not documented in this README. Training-specific results should
instead be inspected through:

- TensorBoard event files,
- training logs,
- metric CSV files,
- saved model metadata,
- evaluation plots.

---

# Repository Structure

```text
mlp_ip_cadelac/
├── cadelac/
│   └── learning/
│       ├── data_scripts/
│       ├── datasets/
│       │   └── panda/
│       ├── models/
│       │   └── context_aware_delan.py
│       ├── trained_models/
│       ├── train_panda.py
│       └── train_panda_v2.py
│
├── scripts/
│   ├── data/
│   │   ├── make_exo_pkl.py
│   │   ├── make_all_exo_pkls.py
│   │   └── fix_exo_pkl_time.py
│   │
│   └── evaluation/
│       ├── evaluate_context_checkpoints.py
│       ├── plot_context_checkpoint_metrics.py
│       ├── plot_zoomed_context_torque_prediction.py
│       └── plot_left_leg_torque_grid.py
│
├── logs/
├── runs/
└── README.md
```

---

# Repository Hygiene

Generated training and evaluation files should normally remain local.

This includes:

```text
logs/
runs/
generated datasets
generated plots
metric CSV files
trained models
checkpoint directories
ZIP packages
Python cache files
editor backup files
```

Before committing changes, inspect the repository:

```bash
git status --short
```

Inspect the files that are staged for the next commit:

```bash
git diff --cached --name-only
```

Check the staged content:

```bash
git diff --cached
```

Check for formatting problems:

```bash
git diff --check
```

The repository should primarily contain reusable source code and documentation.
Generated experiment artifacts should be stored or shared separately.

---

# License

See [`LICENSE`](LICENSE) for the repository license.
