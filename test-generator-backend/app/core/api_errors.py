"""
API errors with a machine-readable code.

Body: {"detail": "<message>", "error": {"code": "<CODE>", "message": "<message>"}}
`detail` keeps every existing frontend working (it shows `detail`); new code switches on `error.code`.
"""

from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

AI_BUSY = "AI_BUSY"
AI_BUSY_MSG = "AI is busy, please try again in a minute."


class ApiError(HTTPException):
    def __init__(self, status_code: int, code: str, message: str, headers: Optional[dict] = None):
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.code = code
        self.message = message


def ai_busy() -> ApiError:
    """Gemini quota / overload after the one allowed retry."""
    return ApiError(503, AI_BUSY, AI_BUSY_MSG, headers={"Retry-After": "60"})


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, headers=exc.headers,
                        content={"detail": exc.message, "error": {"code": exc.code, "message": exc.message}})
