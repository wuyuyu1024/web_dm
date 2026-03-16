# browser-decision-map

Browser-based demo for exploring an interactive inverse projection and decision map on MNIST data. The app loads TensorFlow.js models and a precomputed dataset directly in the browser, then uses D3 to render the map, sample previews, and observation windows.

<p align="center">
  <img
    src="./demo.gif"
    alt="Interactive decision map demo showing the MNIST embedding, inverse projection preview, and observation windows."
    width="960"
  />
</p>

## What the demo does

- Shows MNIST samples as a 2D scatter plot.
- Renders a 100x100 map over the embedding space.
- Predicts an inverse projection image for arbitrary points on the map.
- Visualizes either the decision map, decision confidence, or distance from the initial latent/image state.
- Lets you locally adjust the inverse projection using a selected sample plus a Gaussian neighborhood.
- Provides six observation windows for saving map locations you want to compare.

## Run locally

This repo is a static site. There is no build step and no `package.json`, so you only need to serve the files over HTTP.

Recommended with `pnpm`:

```bash
pnpm dlx serve .
```

Python:

```bash
python3 -m http.server 8000
```

Other common options:

```bash
npx serve .
```

```bash
npx http-server .
```

Then open the local URL printed by the server, for example `http://localhost:8000`.

Notes:

- Open the app through `http://...`, not `file://...`, because the browser needs to fetch `data/mnist/data.json` and the TensorFlow.js model files.
- The app tries to use the TensorFlow.js `webgpu` backend first and falls back to `webgl` if WebGPU is unavailable.
- A Chromium-based browser is the safest choice if you want the best chance of WebGPU support.

## How to use the interface

- Click a scatter point in the main map to show the original MNIST image in the `Real` panel.
- Click anywhere in the map to generate the corresponding inverse projection in the `Inverse Projection` panel.
- Click and drag inside the map to continuously preview inverse projections while moving.
- Use `Map content` to switch between:
  - none
  - distance to the initial `z`
  - distance to the initial reconstructed surface
  - decision map
  - decision map with confidence
- Enable `click data to adjust inverse projection`, then click a sample point to locally modify the inverse projection field around that point.
- Use `Radius` to control the size of the local Gaussian update and `Factor` to control how strongly the latent field is adjusted.
- Click one of the six observation windows, then click on the map to store that location's inverse projection in the selected slot.

## Data and models

The demo ships with one dataset under [`data/mnist`](./data/mnist):

- `data.json`
  - `X`: 3,000 MNIST images, each flattened to 784 values
  - `X2d`: 2D coordinates for those 3,000 samples
  - `z`: 16D latent vectors for those samples
  - `label`: class labels for the 3,000 samples
  - `XY`: a 100x100 evaluation grid flattened to 10,000 2D points
  - `z_of_XY`: 16D latent vectors for the grid points
  - `GRID`: `100`
  - `padding`: `0.05`
- `clf_web/`: TensorFlow.js classifier model used to compute decision regions.
- `Pinv_web_double/`: TensorFlow.js inverse projection model used to reconstruct images from 2D position plus latent code.

## Project structure

- [`index.html`](./index.html): main app shell and external CDN dependencies.
- [`js/main.js`](./js/main.js): startup flow, backend selection, data/model loading, SVG setup, and image rendering helpers.
- [`js/mapholder.js`](./js/mapholder.js): main interaction logic and visualization state management.
- [`js/KNNRegressor.js`](./js/KNNRegressor.js): k-nearest-neighbor regressor used to estimate latent vectors from 2D positions.
- [`css/gridlayout.css`](./css/gridlayout.css): current page layout and widget styling.
- [`index_old.html`](./index_old.html): older layout kept in the repo for reference.

## Implementation notes

- External libraries are loaded from CDNs in `index.html`:
  - Bootstrap 5
  - D3 v7
  - TensorFlow.js
  - TensorFlow.js WebGPU backend
- The app is self-contained aside from those CDN dependencies.
- Some visible text in the UI still reflects prototype-stage wording and placeholders.
