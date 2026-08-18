# PINN Prototype: Compressible Flow Through a Laval Nozzle

This repository implements a physics-informed neural network for the steady compressible flow through an axisymmetric Laval nozzle using the compressible Euler equations.

---

## What the project does

- Trains a PINN to solve the steady compressible Euler equations in a Laval nozzle
- Enforces conservation laws through physics-informed loss terms
- Generates static velocity-magnitude plots for multiple nozzle geometries
- Serves a small FastAPI/WebSocket dashboard for live visualization

---

## PINNs in a nutshell
A Physics-Informed Neural Network (PINN) is a special type of neural network. In a classical neural network, the network prediction ($y_{\mathrm{pred}}$) is compared against training data ($y_{\mathrm{train}}$). The optimizer minimizes an error (loss) function, for example

$$
\mathrm{erf}\left(y_{\mathrm{pred}}-y_{\mathrm{train}}\right),
$$

where $\mathrm{erf}$ denotes a suitable error measure such as the squared error or mean squared error.

A PINN can be trained in the same way as a classical neural network, but it can also approximate the solution of a differential equation. Instead of relying solely on training data, the governing differential equation is incorporated into the loss function. In contrast to classical numerical methods, the required spatial and temporal derivatives are obtained by automatic differentiation of the neural network rather than discretization-based approximations.

Writing the differential equation in residual form, where $\phi$ is the unknown field,

$$
\mathcal{R}(\phi)=0,
$$

the exact solution satisfies this equation everywhere in the computational domain. During training, the neural network predicts an approximate solution, which generally does not satisfy the equation exactly. Evaluating the residual with the network prediction yields $\mathrm{Loss}_{\mathrm{inner}}$. Additional loss terms enforce the boundary and initial conditions. The total loss is then formed as a weighted sum of the individual contributions,

$$
\mathrm{Loss}_{\mathrm{total}} =
w_{\mathrm{inner}}
\mathrm{Loss}_{\mathrm{inner}} +
w_{\mathrm{boundary}}
\mathrm{Loss}_{\mathrm{boundary}} +
w_{\mathrm{data}}
\mathrm{Loss}_{\mathrm{data}} +\cdots,
$$

where the weights control the relative importance of the different objectives during training. In principle, a PINN can approximate the solution of a differential equation using only the governing equations together with the boundary and initial conditions. If additional data are available, they can be incorporated through the data loss to improve the solution or accelerate convergence (see $\mathrm{Loss}_{\mathrm{data}}$).

---

## Physical setup

The flow is modeled as steady, compressible, and inviscid through an axisymmetric Laval nozzle. The network learns the primitive variables density ($\rho$), velocity components ($u$, $v$, $w$), and pressure ($p$) from the conserved-variable form of the Euler equations:

$$
\nabla \cdot (\rho \mathbf{u}) = 0,
$$
$$
\nabla \cdot (\rho \mathbf{u} \mathbf{u} + p \mathbf{I}) = 0,
$$
$$
\nabla \cdot (\rho \mathbf{u} E) = 0,
$$

where $E$ is the total specific energy. The nozzle geometry is parametrized by the throat ratio and is discretized using Cartesian coordinates $(x, y, z)$.

**Boundary conditions:**
- **Inlet:** Uniform primitive variables (density, velocity, pressure)
- **Wall:** Slip boundary condition ($\mathbf{u} \cdot \mathbf{n} = 0$)
- **Outlet:** Pressure fixed; density and velocity components use zero-gradient conditions

---

## Repository layout

```text
.
├── api.py                      # FastAPI app and WebSocket streaming endpoint
├── get_stream_data.py          # Simulation data generation in JSON format
├── index.html                  # Simple browser-based dashboard
├── make_plot_todisk.py         # Static plot export to disk
├── pinn.py                     # PINN architecture, collocation points, and training loop
├── predictor.py                # Inference wrapper around the trained model
├── train.py                    # Training CLI entry point
├── model/                      # PINN model will be written here
├── results/                    # Output plots
└── README.md                   # This file
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

If you have a CUDA-capable GPU and want to use it, install the appropriate version of PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### Training the PINN

To train the model:

```bash
python train.py
```

This will train the PINN and save the model to `model/model.pth`.

### Generating plots

To generate static velocity plots and save them to the `results/` directory:

```bash
python make_plot_todisk.py
```

### Running the live dashboard

In one terminal, start the FastAPI server:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

In another terminal, start a simple HTTP server to serve the frontend:

```bash
python -m http.server 8001
```

Then open your browser to `http://localhost:8001` to view the live dashboard.

---

## Notes

- The PINN training requires automatic differentiation through the neural network, which is handled by PyTorch.
- GPU acceleration is supported and recommended for faster training.
- The model checkpoints are saved in the `model/` directory.
- Visualization outputs are stored in the `results/` directory.