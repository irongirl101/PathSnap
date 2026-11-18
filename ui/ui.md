# Browser UI

`index.html` is served by FastAPI at `/`; it has no build step or third-party frontend dependency. It offers a JSON trace input, map-match action, SVG preview, and server-side evaluation action. It calls same-origin `/api/v1` endpoints, so a LAN client needs only the server laptop's address and port.

The preview draws submitted observations and any predicted points returned by the matcher. The route arc sequence is shown as text because graph geometry rendering depends on the network artifact exposed by the model/HMM integration.
