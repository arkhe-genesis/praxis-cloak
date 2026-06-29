"""cloak backend.

A thin FastAPI service that wraps the local scrub engine (the installable `cloak`
library) and a cloud ChatTransport, exposing the privacy chat loop to the React frontend
over localhost and serving the built frontend itself. It owns no model logic of its own —
scrub and rehydrate are reused from the library verbatim; only the transport and the
local persistence live here. Cloud access is bring-your-own-key (BYOK).
"""
