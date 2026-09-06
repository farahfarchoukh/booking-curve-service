#!/bin/sh
# `exec` — not just running the command — replaces this shell process
# with uvicorn rather than forking it as a child. That's what makes SIGTERM
# from `docker stop` reach uvicorn directly for a graceful shutdown instead
# of hitting this shell (which doesn't forward signals) and forcing a hard
# kill after the grace period. Docker's own CMD linter flags shell-form CMD
# with an env-var default (`${WEB_CONCURRENCY:-1}`) for exactly this
# reason; this script is what lets the Dockerfile use JSON-array CMD (no
# shell wrapping) while still getting the env-var default.
exec uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers "${WEB_CONCURRENCY:-1}"
