# Neural Predictive Control

---
## Description

This project implements an **orbital control algorithm** based on **Model Predictive Control (MPC)**, which employs a Transformer-based **neural network** as the predictive model within the optimization process.

Specifically, the neural network takes as input the **current spacecraft state** along with the **guessed control actions** (velocity increments) provided by the optimizer, and outputs the **predicted trajectory** over a given future time horizon.

---

## Installation

This project is **containerized using Docker**, allowing you to set up the environment without installing any local dependencies other than Docker itself.

### Prerequisites

- [**Docker**](https://www.docker.com/) (Desktop or Engine)

Make sure Docker is running before building the image.

---

### Steps

#### 1. Build the Docker image

The provided Dockerfile defines a **Python 3.12** environment and installs all required dependencies using **uv**.

```bash
docker build -t pinn-mpc -f Dockerfile .
```

#### 2. Run a container

Start an interactive container session and mount the project directory inside it:

```bash
docker run --gpus all --rm -it -v "${PWD}:/workspace" -w /workspace pinn-mpc bash
```

#### 3. Install Python dependencies

Inside the container, synchronize the environment:

```bash
uv sync
```
---

#### 4. Git LFS
To properly download files managed with Git LFS, make sure to run the following command after installing Git LFS:

```bash
git lfs install
```

---

## Run the Simulation  
Once the environment is ready, execute the main script inside the container.  

If you want to run the simulation for the **first operational scenario**, execute:  

```bash  
uv run python MPC_main_asteroid_1.py  
```  

For the **second operational scenario**, execute:  

```bash  
uv run python MPC_main_asteroid_2.py  
```  

---  

### Script Parameters  

Inside these scripts, several scenario parameters can be modified.  

#### Model Selection  

- `SCENARIO`: select the predictive model used inside the MPC.  
  - `"Neural_Network"` → uses the neural network predictive model  
  - `"Linear"` → uses the linear predictive model  

#### Data Saving  

- `SAVE`: if set to `True`, the following data will be saved in the `evaluation_results` directory:  
  - The spacecraft trajectory  
  - The executed control inputs  

#### Visualisation  

- `VISUALISATION`: controls the generation of a simulation video.  
  - If set to **true**, a video of the trajectory tracking simulation will be \*\*saved\*\*.  
  - It also enables **live plotting during execution**.  

> **Note:** Live plotting is not visible when running inside a Docker container.  

---  

### Simulation Time  

The `SIMULATION_TIME` parameter defines the number of time steps for which the simulation will run.  

Time step duration depends on the selected scenario:  

- **Scenario 1:** 5 minutes per time step  
- **Scenario 2:** 1 minute per time step  

---  

### Spin Axis Configuration  

In the **SPIN AXIS** section, it is possible to modify the inclination of the real asteroid spin axis with respect to the body-frame **z-axis**.  

This is done by specifying:  

- **Tilt angle** (in degrees)  
- **Azimuth angle** (in degrees)  

---  

### Asteroid Physical Parameters  

In the **Constants asteroid** section, you can modify the physical parameters of the asteroid.  

- The **true** parameters are used to model the real environment.  
- The **LF** parameters are used to model the onboard spacecraft dynamics when the **Linear** predictive model is selected.  

---  

### MPC Cost Function Weights  

The **Weights** section allows modification of the weighting matrices used in the MPC cost function:  

- **Q** → penalty on position and velocity tracking errors  
- **R** → penalty on control effort  
- **Z** → terminal state penalty  

The matrix **Z** is not defined explicitly. Instead, it is computed as:  

```text  
Z = beta_z * Q  
```  

where `beta_z` is the scaling factor specified in the configuration file.  

---

---

## Training  

The neural networks used in the two operational scenarios can be trained using the scripts:  

- `NET_training_from_zero_asteroid_1.py`  
- `NET_training_from_zero_asteroid_2.py`  

Select the script corresponding to the operational scenario of interest.  

To run the training, enter the appropriate **`Training`** directory. For example:  

```bash  
cd Training_asteroid_1  
```  

Then execute:  

```bash  
uv run python NET_training_from_zero_asteroid_1.py  
```  

---  

### Model evaluation  

The script `evaluation_model.py` allows computing the performance of the trained network in approximating the spacecraft dynamics.  

The evaluation is performed using the **test dataset** contained in the `train_test_traj` directory.  

---  

## Sensitivity analysis  

By running the script `test_on_scenarios.py`, it is possible to perform a sensitivity analysis on the control performance of the algorithm with respect to the asteroid physical parameters:  

- Density  
- Rotation velocity  
- Inclination of the spin axis  

The script takes as input the file `scenarios_parameters.npz`, which contains different possible scenarios characterized by varying asteroid parameters.  

These parameters are sampled within the following ranges:  

- Density: [-30, +30] %  
- Rotation period: [-5, +5] %  
- Tilt angle: [0, 6] degrees  
- Azimuth angle: [0, 360] degrees 

The scenarios are generated using the script:  

`generation_scenarios_parameters.py`  

The output of the sensitivity analysis is the file:  

`Parameters_by_scenario.npy`  

The results can be visualized using the script:  

`post_processing_seaborn.py`  

---  

## Monte Carlo analysis  

By executing the script `Montecarlo_analysis.py`, it is possible to perform a Monte Carlo campaign considering uncertainties sampled from uniform distributions characterized by the same ranges used in the sensitivity analysis.  

The results can be visualized by running:  

`post_processing.py`  

---  

## Processor-in-the-Loop (PIL)  

The `PIL` directory contains the scripts required to perform the Processor-in-the-Loop simulation.  

In particular:  

- `environment.py` must be executed on the laptop.  
- `on_board_online_learning.py` (or `on_board_online_learning_physics.py`) must be executed on the Jetson device, simulating the onboard spacecraft software.  

The interaction between the two platforms is implemented through a **TCP/IP socket-based communication interface**.  

It is also possible to simulate the interaction between the two processes on the same laptop.  

In this configuration:  

- `on_board_online_learning.py` (or `on_board_online_learning_physics.py`) acts as a **Server**.  
- `environment.py` acts as a **Client**, sending the spacecraft state to the server.  

---  

### Online learning configuration  

Within these scripts, it is possible to simulate the **Online Learning** process.  

In the **Online learning** section of the code, several parameters defining the learning behavior can be configured (their meaning is documented directly in the code).  

Inside the **Parameters** section, the following variables can be defined:  

- **EARLY_STOPPING** → enables the early stopping mechanism (stops the online learning process)  
- **PATIENCE** → number of time steps without improvement in the MPE before triggering early stopping  
- **INFERENCE_TIME** → measures the neural network inference time during execution  
- **SOLVE_OPT_TIME** → measures the time required to solve the optimization problem  
- **ONLINE_TRAINING_TIME** → measures the time required to perform one online learning update step  

> **Note:** in `on_board_online_learning_physics.py`, the physics-based loss term is also incorporated into the online learning process. In the **Physics** section, it is possible to specify parameters related to the online estimation of the asteroid physical properties.

---
## Associated Publication

This repository contains the implementation of the Transformer-based Neural Predictive Control framework presented in
the following paper and is released to support the reproducibility of the published results.

> T. Cesarini and S. Silvestrini,
> *Transformer-Based Neural Predictive Control for Small-Body Proximity Operations*, IEEE World Congress on Computational Intelligence, Maastricht, 2026 .


