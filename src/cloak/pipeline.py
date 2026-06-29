"""Default model names for the cloak pipeline.

The library intentionally ships only these constants (no orchestration logic); the
end-to-end harness lives in the eval repo. `DEFAULT_LOCAL_MODEL` is the on-device span
finder served by Ollama; `DEFAULT_CLOUD_MODEL` is the cloud model the protected message
is forwarded to.
"""

DEFAULT_LOCAL_MODEL = "praxis/spanfinder-3b"
DEFAULT_CLOUD_MODEL = "claude-sonnet-4-6"
