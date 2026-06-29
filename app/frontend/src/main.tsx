import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import App from "./App.tsx"

// Open-source, no-auth build: render the app directly (BYOK — the user supplies their
// own provider key in Settings; there is no sign-in gate).
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
