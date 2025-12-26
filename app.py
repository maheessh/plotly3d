import base64
import io
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch

import plotly.graph_objects as go
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc 
from PIL import Image



EXTERNAL_STYLESHEETS = [dbc.themes.CYBORG]

RANDOM_SEED_DEFAULT = 42
NUM_ANCHORS_DEFAULT = 6
MAX_DISPLAY_POINTS = 2000
PCA_COMPONENTS = 3

IMAGE_DIR = Path(__file__).resolve().parent / "data" / "images"

# ==============================
# Load Results

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_PATH = DATA_DIR / "results.pt"

try:
    data = torch.load(RESULTS_PATH, map_location="cpu")
    results = data["results"]
    head_details = data.get("head_details", {})
except FileNotFoundError:
    print(f"WARNING: {RESULTS_PATH} not found. Using dummy data for demo.")
    results = [{"vector": np.random.rand(128), "prediction": [0.1]*10} for _ in range(500)]
    head_details = {}

TOTAL = len(results)

# Build embedding matrix
all_embeddings = np.asarray(
    [np.asarray(r["vector"], dtype=np.float32) for r in results],
    dtype=np.float32
)

# Normalize
norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1.0
all_embeddings = all_embeddings / norms
D = all_embeddings.shape[1]

# PCA
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED_DEFAULT)
coords = pca.fit_transform(all_embeddings).astype(np.float32)

# Sphere Normalization
coord_norms = np.linalg.norm(coords, axis=1, keepdims=True)
coord_norms[coord_norms == 0] = 1.0
coords = coords / coord_norms

# Helpers
def apply_brightness(hex_color: str, sim: float):
    rgb = np.array(mcolors.to_rgb(hex_color))
    brightness = 0.25 + 0.75 * ((sim + 1) / 2)
    return tuple((rgb * brightness).clip(0, 1))

def try_find_image_path(idx: int) -> Path | None:
    if not IMAGE_DIR.exists():
        return None
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = IMAGE_DIR / f"{idx}{ext}"
        if p.exists():
            return p
    return None

@lru_cache(maxsize=512)
def image_file_to_data_uri(path_str: str) -> str:
    try:
        p = Path(path_str)
        img = Image.open(p).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""

def build_figure(display_indices: np.ndarray, anchor_indices: np.ndarray):
    """Create a polished Plotly figure that blends with the UI."""
    disp_emb = all_embeddings[display_indices]
    anch_emb = all_embeddings[anchor_indices]
    sim_matrix = disp_emb @ anch_emb.T
    closest_anchor = np.argmax(sim_matrix, axis=1)
    max_sim = np.max(sim_matrix, axis=1)

    anchor_colors = list(mcolors.TABLEAU_COLORS.values())
    if len(anchor_colors) < len(anchor_indices):
        anchor_colors = (anchor_colors * ((len(anchor_indices) // len(anchor_colors)) + 1))[:len(anchor_indices)]
    else:
        anchor_colors = anchor_colors[:len(anchor_indices)]

    data_colors = [
        apply_brightness(anchor_colors[int(closest_anchor[i])], float(max_sim[i]))
        for i in range(len(display_indices))
    ]

    disp_xyz = coords[display_indices]
    anch_xyz = coords[anchor_indices]

    fig = go.Figure()
    
    # Custom Data mapping: [index, anchor_idx, similarity]
    customdata = np.stack([display_indices, closest_anchor, max_sim], axis=1)

    fig.add_trace(go.Scatter3d(
        x=disp_xyz[:, 0], y=disp_xyz[:, 1], z=disp_xyz[:, 2],
        mode="markers",
        name="Data Points",
        marker=dict(size=4, color=data_colors, opacity=0.8),
        customdata=customdata,
        hovertemplate="<b>Idx:</b> %{customdata[0]}<br><b>Sim:</b> %{customdata[2]:.3f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter3d(
        x=anch_xyz[:, 0], y=anch_xyz[:, 1], z=anch_xyz[:, 2],
        mode="markers+text",
        name="Anchors",
        marker=dict(size=12, color=anchor_colors, symbol="diamond", line=dict(width=2, color='white')),
        text=[f"A{i}" for i in range(len(anchor_indices))],
        textposition="top center",
        textfont=dict(color="white", size=14),
        customdata=np.stack([np.arange(len(anchor_indices)), anchor_indices], axis=1),
        hovertemplate="<b>Anchor A%{customdata[0]}</b><br>ID: %{customdata[1]}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0),
        height=750,
        scene=dict(
            xaxis=dict(showticklabels=False, title='', showgrid=True, gridcolor='#333'),
            yaxis=dict(showticklabels=False, title='', showgrid=True, gridcolor='#333'),
            zaxis=dict(showticklabels=False, title='', showgrid=True, gridcolor='#333'),
            aspectmode="cube",
            bgcolor='rgba(0,0,0,0)' 
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            yanchor="top", y=0.95, xanchor="left", x=0.05,
            bgcolor="rgba(0,0,0,0.5)", bordercolor="#444", borderwidth=1
        ),
        uirevision="keep_camera",
        clickmode='event+select'
    )
    return fig

def make_anchor_indices(seed: int, num_anchors: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(TOTAL, size=num_anchors, replace=False)

def make_display_indices(seed: int, num_points: int, anchor_indices: np.ndarray) -> np.ndarray:
    num_points = int(np.clip(num_points, 10, min(MAX_DISPLAY_POINTS, TOTAL - len(anchor_indices))))
    rng = np.random.default_rng(seed + 999)
    mask = np.ones(TOTAL, dtype=bool)
    mask[anchor_indices] = False
    eligible = np.where(mask)[0]
    return rng.choice(eligible, size=num_points, replace=False)

# UI COMPONENTS

def create_control_card(label, control, help_text=None):
    return dbc.Card(
        [
            dbc.CardHeader(label, className="text-uppercase small fw-bold text-muted"),
            dbc.CardBody(
                [
                    control,
                    html.Small(help_text, className="text-muted mt-2 d-block") if help_text else None
                ],
                className="p-3"
            )
        ],
        className="h-100 border-secondary shadow-sm"
    )

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.I(className="bi bi-diagram-3-fill fs-3 text-info me-2")),
                    dbc.Col(dbc.NavbarBrand("Embedding Explorer", className="ms-2 fw-bold fs-4")),
                ],
                align="center",
                className="g-0",
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.NavItem(html.Span(f"N={TOTAL} | Dim={D}", className="text-muted small align-self-center")),
                    ],
                    className="ms-auto",
                    navbar=True,
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="mb-4 border-bottom border-secondary"
)

controls = dbc.Row(
    [
        dbc.Col(
            create_control_card(
                "Display Points",
                dcc.Slider(
                    id="num-points",
                    min=50,
                    max=min(MAX_DISPLAY_POINTS, TOTAL - 10),
                    step=50,
                    value=300,
                    marks={50: '50', 1000: '1k', min(MAX_DISPLAY_POINTS, TOTAL - 10): 'Max'},
                    className="pt-2"
                ),
                "Number of points to visualize"
            ),
            md=4,
        ),
        dbc.Col(
            create_control_card(
                "Anchors",
                dcc.Slider(
                    id="num-anchors",
                    min=2, max=12, step=1, value=NUM_ANCHORS_DEFAULT,
                    marks={2: '2', 6: '6', 12: '12'},
                    className="pt-2"
                ),
                "Reference points for similarity"
            ),
            md=4,
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Sampling", className="text-uppercase small fw-bold text-muted"),
                    dbc.CardBody(
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Seed"),
                                dbc.Input(id="seed", type="number", value=RANDOM_SEED_DEFAULT),
                                dbc.Button(
                                    [html.I(className="bi bi-shuffle me-2"), "Resample"],
                                    id="resample", color="info", n_clicks=0
                                ),
                            ]
                        ),
                        className="p-3"
                    )
                ],
                className="h-100 border-secondary shadow-sm"
            ),
            md=4,
        ),
    ],
    className="g-3 mb-4"
)

# APP LAYOUT
app = Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)
app.title = "Embedding Explorer"

app.layout = dbc.Container(
    [
        dcc.Store(id="store-indices"),
        navbar,
        controls,
        dbc.Row(
            [
                # GRAPH COLUMN
                dbc.Col(
                    dbc.Card(
                        [
                            dcc.Loading(
                                id="loading-graph",
                                type="cube",
                                color="#17a2b8",
                                children=dcc.Graph(
                                    id="graph-3d",
                                    config={"displayModeBar": True, "displaylogo": False},
                                    style={"height": "750px"}
                                )
                            )
                        ],
                        className="border-secondary shadow-sm bg-dark"
                    ),
                    lg=9, md=12, className="mb-3"
                ),
                
                # INSPECTOR COLUMN
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [html.I(className="bi bi-search me-2"), "Inspector"],
                                className="bg-transparent border-secondary fw-bold"
                            ),
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            html.Img(
                                                id="preview-img",
                                                src="",
                                                className="img-fluid rounded border border-secondary mb-3",
                                                style={"display": "none", "objectFit": "cover"}
                                            ),
                                            html.Div(id="preview-text", className="small")
                                        ],
                                        style={"minHeight": "200px"}
                                    )
                                ],
                                style={"maxHeight": "750px", "overflowY": "auto"}
                            ),
                            dbc.CardFooter(
                                html.Small("Click on a node to lock details.", className="text-muted"),
                                className="border-secondary"
                            )
                        ],
                        className="h-100 border-secondary shadow-sm bg-dark"
                    ),
                    lg=3, md=12, className="mb-3"
                ),
            ],
            className="g-3"
        )
    ],
    fluid=True,
    className="pb-5",
    style={"minHeight": "100vh", "backgroundColor": "#000"}
)


# CALLBACKS

@app.callback(
    Output("store-indices", "data"),
    Input("resample", "n_clicks"),
    State("num-points", "value"),
    State("num-anchors", "value"),
    State("seed", "value"),
)
def resample_indices(n_clicks, num_points, num_anchors, seed):
    seed = int(seed) if seed is not None else RANDOM_SEED_DEFAULT
    num_anchors = int(num_anchors) if num_anchors is not None else NUM_ANCHORS_DEFAULT
    anchor_idx = make_anchor_indices(seed, num_anchors)
    display_idx = make_display_indices(seed, num_points, anchor_idx)
    return {
        "anchor_indices": anchor_idx.tolist(),
        "display_indices": display_idx.tolist(),
        "seed": seed,
    }

@app.callback(
    Output("graph-3d", "figure"),
    Input("store-indices", "data"),
    prevent_initial_call=True
)
def update_figure_callback(store_data):
    if not store_data:
        return no_update
    anchor_indices = np.array(store_data["anchor_indices"], dtype=int)
    display_indices = np.array(store_data["display_indices"], dtype=int)
    return build_figure(display_indices, anchor_indices)


# CHANGED: ClickData Callback
@app.callback(
    Output("preview-img", "src"),
    Output("preview-img", "style"),
    Output("preview-text", "children"),
    Input("graph-3d", "clickData"), # Changed from hoverData to clickData
)
def update_preview(clickData):
    default_text = html.Div([
        html.Div(className="bi bi-hand-index-thumb fs-1 text-secondary mb-3"),
        html.P("Click a node to view details", className="text-muted"),
    ], className="text-center mt-5")
    
    if not clickData or "points" not in clickData or not clickData["points"]:
        return "", {"display": "none"}, default_text

    p = clickData["points"][0]
    curve = p.get("curveNumber", 0)

    # 0 = Data Points, 1 = Anchors
    if curve == 0:
        idx_raw, closest_anchor, sim = p.get("customdata", [None, None, None])
        idx = int(idx_raw)
        pred = results[idx]["prediction"]
        
        type_badge = dbc.Badge("Data Point", color="primary", className="mb-2")
        details = [
            html.Div([html.Strong("Index: "), f"{idx}"]),
            html.Div([html.Strong("Similarity: "), f"{float(sim):.4f}"]),
            html.Div([html.Strong("Closest: "), dbc.Badge(f"A{int(closest_anchor)}", color="secondary", className="ms-1")]),
            html.Hr(className="my-2 border-secondary"),
            html.Small("Prediction Head (First 8):", className="text-muted"),
            html.Pre(str(pred[:8]), className="bg-black p-2 rounded text-info small mt-1"),
        ]
        target_idx = idx

    else:
        cd = p.get("customdata", [None, None])
        anchor_id = int(cd[0])
        idx = int(cd[1])
        pred = results[idx]["prediction"]
        
        type_badge = dbc.Badge(f"Anchor A{anchor_id}", color="warning", text_color="dark", className="mb-2")
        details = [
            html.Div([html.Strong("Original Index: "), f"{idx}"]),
            html.Hr(className="my-2 border-secondary"),
            html.Small("Prediction Head (First 8):", className="text-muted"),
            html.Pre(str(pred[:8]), className="bg-black p-2 rounded text-warning small mt-1"),
        ]
        target_idx = idx

    # Image Handling
    img_path = try_find_image_path(target_idx)
    if img_path:
        src = image_file_to_data_uri(str(img_path))
        style = {"display": "block", "width": "100%", "borderRadius": "8px"}
    else:
        src = ""
        style = {"display": "none"}
        details.append(html.Div("No image found.", className="text-muted small fst-italic mt-2"))

    return src, style, html.Div([type_badge, html.Div(details)])


if __name__ == "__main__":
    app.run(debug=True)