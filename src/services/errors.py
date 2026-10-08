"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Answer a rejected request body with 422 and no echo of the input.

    FastAPI's default handler echoes the rejected ``input``, which crashes the
    JSON encoder (HTTP 500) when the body held NaN or Infinity.

    Args:
        _: The failed request.
        exc: Validation error raised for the request.

    Returns:
        A 422 response whose ``detail`` lists the ``loc``, ``msg`` and ``type``
        of every error.
    """
    errors = [
        {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


def register_validation_handler(app: FastAPI) -> None:
    """Install the NaN-safe request validation handler on a service.

    Args:
        app: Service application to configure.
    """
    app.add_exception_handler(RequestValidationError, validation_error_handler)
