from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import asyncio
import torch

from pinn import PINN
from predictor import Predictor
from get_stream_data import generate_surface_velocity_data

# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model = PINN(hidden_dim=64, n_layers=2).to(DEVICE)
checkpoint = torch.load("model/model.pth", map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

predictor = Predictor(model=model, device=DEVICE)


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


simulation_parameters = {
    "throat_ratio": 0.3,
}


# ---------------------------------------------------------
# Optional REST endpoint
# ---------------------------------------------------------

from pydantic import BaseModel


class SimulationConfig(BaseModel):
    throat_ratio: float
    """
    Pydantic model for simulation configuration.

    Parameters
    ----------
    throat_ratio : float
        Angular velocity used by the simulation stream.
    """


@app.post("/simulation/config")
async def update_config(
    config: SimulationConfig,
):
    """
    REST endpoint to update simulation parameters.

    Parameters
    ----------
    config : SimulationConfig
        Configuration payload containing the updated `throat_ratio` value.

    Returns
    -------
    dict
        A status dictionary confirming the update and echoing the new `throat_ratio`.
    """

    simulation_parameters["throat_ratio"] = config.throat_ratio

    return {
        "status": "updated",
        "throat_ratio": config.throat_ratio,
    }


# ---------------------------------------------------------
# WebSocket stream
# ---------------------------------------------------------


@app.websocket("/graph-stream")
async def graph_stream(
    websocket: WebSocket,
):
    """
    WebSocket endpoint that streams velocity field data and accepts parameter updates.

    This endpoint accepts a WebSocket connection, then runs two background
    tasks: a `sender` that periodically computes velocity data via
    `generate_velocity_data` and sends it to the client, and a `receiver` that
    listens for incoming JSON messages to update simulation parameters
    (currently `throat_ratio`). Both tasks run until the client disconnects.

    Parameters
    ----------
    websocket : WebSocket
        The active WebSocket connection to the client.

    Returns
    -------
    None. Streams JSON messages to the connected client and processes incoming
    parameter updates in-place.
    """

    await websocket.accept()

    last_throat_ratio = None
    last_data = None

    async def sender():
        nonlocal last_throat_ratio, last_data

        while True:
            throat_ratio = simulation_parameters["throat_ratio"]

            data = await asyncio.to_thread(
                generate_surface_velocity_data,
                predictor,
                throat_ratio=throat_ratio,
            )

            # Discard outdated result if throat_ratio changed while computation ran
            if throat_ratio != simulation_parameters["throat_ratio"]:
                continue

            last_throat_ratio = throat_ratio
            last_data = data

            await websocket.send_json(data)

            await asyncio.sleep(0.01)

    async def receiver():
        """
        Receive JSON messages from the WebSocket and update simulation parameters.

        Expected message format for parameter updates:
        `{"type": "parameterUpdate", "throat_ratio": <value>}`.

        Returns
        -------
        None.
        """

        while True:

            message = await websocket.receive_json()

            if message.get("type") == "parameterUpdate":
                value = float(message["throat_ratio"])
                simulation_parameters["throat_ratio"] = min(0.4, max(0.2, value))

    sender_task = asyncio.create_task(sender())

    receiver_task = asyncio.create_task(receiver())

    try:

        await asyncio.gather(
            sender_task,
            receiver_task,
        )

    except WebSocketDisconnect:

        print("WebSocket client disconnected")

    except Exception as exc:

        print(f"WebSocket error: {exc}")

    finally:

        sender_task.cancel()
        receiver_task.cancel()

        await asyncio.gather(
            sender_task,
            receiver_task,
            return_exceptions=True,
        )
