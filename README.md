# PINN Prototype: Compressible Flow Through a Laval Nozzle

This repository implements a physics-informed neural network in PyTorch for the steady compressible flow through an axisymmetric Laval nozzle using the compressible Euler equations.

---

## What the project does

- Trains a PINN to solve the steady compressible Euler equations in a Laval nozzle with variable geometry
- Enforces conservation laws through physics-informed loss terms
- Generates various plots
- Serves a small FastAPI/WebSocket dashboard for live visualization

---


## Physical set-up

The flow is modeled as steady, compressible, and inviscid through an axisymmetric Laval nozzle. The network learns the primitive variables density ($\rho$), velocity components ($u$, $v$, $w$), and pressure ($p$) from the conserved-variable form of the Euler equations:

$$
\nabla \cdot (\rho 𝐮) = 0,
$$
$$
\nabla \cdot (\rho 𝐮 \otimes 𝐮 + p \mathbf{I}) = 0,
$$
$$
\nabla \cdot [(E+p) 𝐮] = 0,
$$

where $E$ is the total specific energy. The nozzle geometry is parametrized by the throat ratio and is discretized using Cartesian coordinates $(x, y, z)$.

**Boundary conditions:**
- **Inlet:** Uniform primitive variables (density, velocity, pressure)
- **Wall:** Slip boundary condition ($𝐮 \cdot \mathbf{n} = 0$)
- **Outlet:** zero-gradient conditions for all primitive variables

---

## Repository layout

```text
.
├── APP/
│   ├── api.py                   --> FastAPI app and WebSocket streaming endpoint
│   ├── get_stream_data.py       --> Simulation data generation in JSON format
│   ├── index.html               --> Browser-based live dashboard
│   └── lib/                     --> Local VTK.js and Plotly JavaScript libraries
├── CONDITIONS/
│   ├── boundary_conditions.py   --> Inlet, wall, and outlet conditions
│   └── flow_quantities.py       --> Flow and thermodynamic quantities
├── DIRS/
│   └── dirs.py                  --> Project directory and model-path definitions
├── DOC/
│   └── PINN_description.pdf     --> Documentation providing a detailed description of the PINN
├── GEOMETRY/
│   ├── collocate.py             --> Collocation-point generation
│   ├── geometry.py              --> Laval-nozzle geometry
│   └── probes.py                --> Centerline, slice, and surface probes
├── MISC/
│   ├── calc_ref_states.py       --> Reference-state calculations
│   └── make_plot_todisk.py      --> Static plot export to disk
├── PINN/
│   ├── derivatives.py           --> Automatic-differentiation helpers
│   ├── pinn.py                  --> PINN architecture and physics-informed loss
│   └── predictor.py             --> Inference wrapper around the trained model
├── TRAIN/
│   ├── monitor.py               --> Training monitoring utilities
│   ├── retrain.py               --> Model retraining entry point
│   └── train.py                 --> Training CLI entry point
├── model/
│   └── model.pth                --> Current trained model checkpoint (see DIRS/dirs.py)
├── model_pretrained/
│   └── model.pth                --> Pretrained model checkpoint
├── results/                     --> Generated training and probe results (see DIRS/dirs.py)
└── README.md                    --> This file
```

---

## Requirements and setup

The project has been tested with Python 3.10+.

### 1. Create and activate a virtual environment

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install torch numpy matplotlib fastapi "uvicorn[standard]" pydantic
```

---

## Usage

### Training the PINN

To train the model:

```bash
python TRAIN/train.py
```

This will train the PINN and save the model to `model/model.pth`. Alternatively, you can skip the training step and proceed with the pre-trained PINN `model_pretrained/model.pth`. To do so, copy the pre-trained model into `model/`.

### Generating plots

To generate static velocity plots and save them to the `results/` directory:

```bash
python MISC/make_plot_todisk.py
```

### Running the live dashboard

In one terminal, start the FastAPI server:

```bash
python -m uvicorn APP.api:app --host 127.0.0.1 --port 8000
```

In another terminal, start a simple HTTP server to serve the frontend:

```bash
python -m http.server 8001 --directory ./APP
```

Then open your browser to `http://localhost:8001` to view the live dashboard.

---

## Notes

- The PINN training requires automatic differentiation through the neural network, which is handled by PyTorch.
- GPU acceleration is supported but has not been tested due to limitations of the present machine.