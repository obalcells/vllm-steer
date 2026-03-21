# SPDX-License-Identifier: Apache-2.0
"""HTTP endpoints for activation steering.

POST /v1/steer_vector    — set the active steer vector on all workers
DELETE /v1/steer_vector  — clear steering (restore stock behaviour)
"""

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


class SteerVectorBody(BaseModel):
    steer_vector_name: str
    steer_vector_int_id: int
    steer_vector_local_path: str
    scale: float = 1.0
    target_layers: list[int] | None = None
    prefill_trigger_tokens: list[int] | None = None
    generate_trigger_tokens: list[int] | None = None
    algorithm: str = "direct"
    normalize: bool = False
    hook: str = "pre"


def _engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/v1/steer_vector")
async def set_steer_vector(body: SteerVectorBody, raw_request: Request):
    spec: dict[str, Any] = body.model_dump()
    try:
        await _engine_client(raw_request).collective_rpc(
            "set_steer_vector", args=(spec,)
        )
    except Exception as e:
        logger.exception("set_steer_vector failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e
    return JSONResponse(content={"ok": True, "steer_vector": spec})


@router.delete("/v1/steer_vector")
async def clear_steer_vector(raw_request: Request):
    try:
        await _engine_client(raw_request).collective_rpc(
            "set_steer_vector", args=(None,)
        )
    except Exception as e:
        logger.exception("clear_steer_vector failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e
    return JSONResponse(content={"ok": True})


class CaptureBody(BaseModel):
    cmd: str  # "start" or "stop"
    positions: list[int] | None = None
    save_path: str | None = None


@router.post("/v1/steer_capture")
async def steer_capture(body: CaptureBody, raw_request: Request):
    import json as _json
    arg = None
    if body.cmd == "start" and body.positions is not None:
        arg = _json.dumps(body.positions)
    elif body.cmd == "stop" and body.save_path is not None:
        arg = body.save_path
    try:
        results = await _engine_client(raw_request).collective_rpc(
            "steer_capture", args=(body.cmd, arg)
        )
    except Exception as e:
        logger.exception("steer_capture failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e
    return JSONResponse(content={"ok": True, "workers": results})


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
