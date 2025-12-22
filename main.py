import base64
import io
import numpy as np
import torch

import plotly.graph_objects as go
import matplotlib.colors as mcolors

from datasets import load_dataset
from sklearn.decomposition import PCA
from transformers import CLIPModel, CLIPProcessor

from dash import Dash, dcc, html, Input, Output


# CONFIG

NUM_DATA_POINTS = 120
NUM_ANCHORS = 6
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(RANDOM_SEED)



# Helpers

def pil_to_data_uri(pil_img) -> str:
    """Convert PIL image -> data URI for <img src="...">."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def apply_brightness(hex_color: str, sim: float):
    """Make color brighter when similarity is higher. sim assumed in [-1, 1]."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    brightness = 0.25 + 0.75 * ((sim + 1) / 2)  # map [-1,1] -> [0.25,1]
    return tuple((rgb * brightness).clip(0, 1))


# Load dataset (real jewelry images)

ds = load_dataset("sidd707/jewelry-design-dataset", split="train")
total_needed = NUM_ANCHORS + NUM_DATA_POINTS
ds = ds.shuffle(seed=RANDOM_SEED).select(range(total_needed))

pil_images = [item["image"] for item in ds]
anchor_pils = pil_images[:NUM_ANCHORS]
data_pils = pil_images[NUM_ANCHORS:]

# Pre-encode for fast hover preview
anchor_img_src = [pil_to_data_uri(img) for img in anchor_pils]
data_img_src = [pil_to_data_uri(img) for img in data_pils]


# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()


@torch.no_grad()
def compute_clip_embeddings(pil_list):
    inputs = processor(images=pil_list, return_tensors="pt", padding=True).to(DEVICE)
    emb = model.get_image_features(**inputs)  # [N, D]
    emb = emb.detach().cpu().numpy()
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) 
    return emb


all_embeddings = compute_clip_embeddings(pil_images)
anchor_embeddings = all_embeddings[:NUM_ANCHORS]
data_embeddings = all_embeddings[NUM_ANCHORS:]


# Cosine similarity: data vs anchors

similarity_matrix = data_embeddings @ anchor_embeddings.T 
closest_anchor_idx = np.argmax(similarity_matrix, axis=1)
max_similarity = np.max(similarity_matrix, axis=1)


# PCA -> 3D (for visualization) + sphere normalization
pca = PCA(n_components=3, random_state=RANDOM_SEED)
all_3d = pca.fit_transform(np.vstack([anchor_embeddings, data_embeddings]))
all_3d /= np.linalg.norm(all_3d, axis=1, keepdims=True) 

anchor_vectors = all_3d[:NUM_ANCHORS]
data_vectors = all_3d[NUM_ANCHORS:]


# Colors
anchor_colors = list(mcolors.TABLEAU_COLORS.values())[:NUM_ANCHORS]
data_colors = [
    apply_brightness(anchor_colors[closest_anchor_idx[i]], float(max_similarity[i]))
    for i in range(NUM_DATA_POINTS)
]


# Plotly Figure
data_customdata = np.stack(
    [np.arange(NUM_DATA_POINTS), closest_anchor_idx, max_similarity],
    axis=1
)

fig = go.Figure()

# Many points (data)
fig.add_trace(go.Scatter3d(
    x=data_vectors[:, 0],
    y=data_vectors[:, 1],
    z=data_vectors[:, 2],
    mode="markers",
    name="Jewelry Images",
    marker=dict(size=4, color=data_colors, opacity=0.85),
    customdata=data_customdata,
    hovertemplate=(
        "<b>Image Index:</b> %{customdata[0]}<br>"
        "<b>Closest Anchor:</b> A%{customdata[1]}<br>"
        "<b>Cosine Similarity:</b> %{customdata[2]:.3f}<br>"
        "<extra></extra>"
    )
))

# Few points (anchors)
# Put anchor index in customdata as well so hoverData includes it
anchor_customdata = np.arange(NUM_ANCHORS).reshape(-1, 1)

fig.add_trace(go.Scatter3d(
    x=anchor_vectors[:, 0],
    y=anchor_vectors[:, 1],
    z=anchor_vectors[:, 2],
    mode="markers+text",
    name="Anchor Images",
    marker=dict(size=16, color=anchor_colors, symbol="diamond"),
    text=[f"A{i}" for i in range(NUM_ANCHORS)],
    textposition="top center",
    customdata=anchor_customdata,
    hovertemplate=(
        "<b>Anchor:</b> A%{customdata[0]}<br>"
        "<extra></extra>"
    )
))

fig.update_layout(
    template="plotly_dark",
    title="3D Jewelry Image Similarity (CLIP · Cosine · Spherical)",
    height=750,
    scene=dict(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        zaxis=dict(showticklabels=False),
        aspectmode="cube",
    ),
    margin=dict(l=0, r=0, t=60, b=0),
)


# Dash App
app = Dash(__name__)
app.title = "Jewelry Similarity Explorer (Dash)"

app.layout = html.Div(
    style={
        "backgroundColor": "#0E1117",
        "color": "white",
        "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial",
        "padding": "16px",
    },
    children=[
        html.Div(
            "Hover points to preview images. Tooltip shows index, closest anchor, and cosine similarity.",
            style={"opacity": 0.85, "marginBottom": "12px"},
        ),

        html.Div(
            style={"display": "flex", "gap": "16px"},
            children=[
                html.Div(
                    style={"flex": 3, "minWidth": "0"},
                    children=[
                        dcc.Graph(
                            id="graph-3d",
                            figure=fig,
                            clear_on_unhover=False, 
                            config={"displayModeBar": True},
                        ),
                    ],
                ),

                html.Div(
                    style={
                        "flex": 1,
                        "minWidth": "280px",
                        "background": "#111827",
                        "border": "1px solid #1F2937",
                        "borderRadius": "12px",
                        "padding": "12px",
                        "height": "750px",
                        "overflow": "auto",
                    },
                    children=[
                        html.H4("Preview", style={"marginTop": "0px"}),
                        html.Img(
                            id="preview-img",
                            src="",
                            style={
                                "width": "100%",
                                "borderRadius": "10px",
                                "border": "1px solid #1F2937",
                                "display": "none",  
                            },
                        ),
                        html.Div(
                            id="preview-text",
                            style={"marginTop": "10px", "lineHeight": "1.5"},
                            children="Hover a point to preview the image here.",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("preview-img", "src"),
    Output("preview-img", "style"),
    Output("preview-text", "children"),
    Input("graph-3d", "hoverData"),
)
def update_preview(hoverData):
    if not hoverData or "points" not in hoverData or len(hoverData["points"]) == 0:
        return "", {"width": "100%", "borderRadius": "10px", "display": "none"}, "Hover a point to preview the image here."

    p = hoverData["points"][0]

    curve = p.get("curveNumber", 0)

    idx = p.get("pointIndex", p.get("pointNumber", 0))

    if curve == 0:
        # data point
        # customdata = [index, closest_anchor, similarity]
        cd = p.get("customdata", [idx, int(closest_anchor_idx[idx]), float(max_similarity[idx])])
        img_index = int(cd[0])
        anchor_idx = int(cd[1])
        sim = float(cd[2])

        text = [
            html.Div([html.B("Type: "), "Jewelry Image"]),
            html.Div([html.B("Index: "), str(img_index)]),
            html.Div([html.B("Closest Anchor: "), f"A{anchor_idx}"]),
            html.Div([html.B("Cosine Similarity: "), f"{sim:.3f}"]),
        ]

        return (
            data_img_src[img_index],
            {"width": "100%", "borderRadius": "10px", "border": "1px solid #1F2937", "display": "block"},
            text,
        )

    # anchor point
    # customdata = [anchor_idx]
    cd = p.get("customdata", [idx])
    anchor_idx = int(cd[0])

    text = [
        html.Div([html.B("Type: "), "Anchor Image"]),
        html.Div([html.B("Anchor: "), f"A{anchor_idx}"]),
    ]

    return (
        anchor_img_src[anchor_idx],
        {"width": "100%", "borderRadius": "10px", "border": "1px solid #1F2937", "display": "block"},
        text,
    )


if __name__ == "__main__":
    app.run(debug=True)
