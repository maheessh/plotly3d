import os
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import streamlit as st
import torch

from transformers import CLIPProcessor, CLIPModel
from sklearn.decomposition import PCA
from datasets import load_dataset
from streamlit_plotly_events import plotly_events

# ==============================
# CONFIG
# ==============================
NUM_DATA_POINTS = 120
NUM_ANCHORS = 6
RANDOM_SEED = 42

IMG_DIR = "tmp_images"
os.makedirs(IMG_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)
st.set_page_config(layout="wide")

# ==============================
# Load real jewelry dataset
# ==============================
@st.cache_data(show_spinner=True)
def load_images():
    ds = load_dataset("sidd707/jewelry-design-dataset", split="train")

    total_needed = NUM_DATA_POINTS + NUM_ANCHORS
    ds = ds.shuffle(seed=RANDOM_SEED).select(range(total_needed))

    pil_images = [item["image"] for item in ds]

    image_paths = []
    for i, img in enumerate(pil_images):
        path = os.path.abspath(f"{IMG_DIR}/img_{i}.png")
        img.save(path)
        image_paths.append(path)

    return pil_images, image_paths

pil_images, image_paths = load_images()

anchor_image_paths = image_paths[:NUM_ANCHORS]
data_image_paths = image_paths[NUM_ANCHORS:]

# ==============================
# Load CLIP model
# ==============================
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

model, processor = load_clip()

# ==============================
# Image → CLIP embeddings
# ==============================
@st.cache_data(show_spinner=True)
def compute_image_embeddings(_images):
    inputs = processor(images=_images, return_tensors="pt", padding=True)

    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    embeddings = embeddings.cpu().numpy()
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

all_embeddings = compute_image_embeddings(pil_images)

anchor_embeddings = all_embeddings[:NUM_ANCHORS]
data_embeddings = all_embeddings[NUM_ANCHORS:]

# ==============================
# Cosine similarity
# ==============================
similarity_matrix = data_embeddings @ anchor_embeddings.T
closest_anchor_idx = np.argmax(similarity_matrix, axis=1)
max_similarity = np.max(similarity_matrix, axis=1)

# ==============================
# PCA → 3D + sphere normalization
# ==============================
pca = PCA(n_components=3, random_state=RANDOM_SEED)
all_3d = pca.fit_transform(
    np.vstack([anchor_embeddings, data_embeddings])
)

# Normalize PCA output to unit sphere
all_3d /= np.linalg.norm(all_3d, axis=1, keepdims=True)

anchor_vectors = all_3d[:NUM_ANCHORS]
data_vectors = all_3d[NUM_ANCHORS:]

# ==============================
# Colors
# ==============================
anchor_colors = list(mcolors.TABLEAU_COLORS.values())[:NUM_ANCHORS]

def apply_brightness(hex_color, sim):
    rgb = np.array(mcolors.to_rgb(hex_color))
    brightness = 0.25 + 0.75 * ((sim + 1) / 2)
    return tuple((rgb * brightness).clip(0, 1))

data_colors = [
    apply_brightness(anchor_colors[closest_anchor_idx[i]], max_similarity[i])
    for i in range(NUM_DATA_POINTS)
]

# ==============================
# Plotly figure (RENDERED chart)
# ==============================
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=data_vectors[:, 0],
    y=data_vectors[:, 1],
    z=data_vectors[:, 2],
    mode="markers",
    marker=dict(size=4, color=data_colors, opacity=0.85),
    customdata=np.stack([
        np.arange(NUM_DATA_POINTS),
        closest_anchor_idx,
        max_similarity
    ], axis=1),
    hovertemplate=(
        "<b>Image Index:</b> %{customdata[0]}<br>"
        "<b>Closest Anchor:</b> A%{customdata[1]}<br>"
        "<b>Cosine Similarity:</b> %{customdata[2]:.3f}<br>"
        "<extra></extra>"
    ),
    name="Jewelry Images"
))

fig.add_trace(go.Scatter3d(
    x=anchor_vectors[:, 0],
    y=anchor_vectors[:, 1],
    z=anchor_vectors[:, 2],
    mode="markers+text",
    marker=dict(size=16, color=anchor_colors, symbol="diamond"),
    text=[f"A{i}" for i in range(NUM_ANCHORS)],
    textposition="top center",
    name="Anchor Images"
))

fig.update_layout(
    template="plotly_dark",
    title="3D Jewelry Image Similarity (CLIP · Cosine · Spherical)",
    height=700,
    scene=dict(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        zaxis=dict(showticklabels=False),
        aspectmode="cube"
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)

# ==============================
# UI
# ==============================
st.markdown("""
## 💎 3D Jewelry Image Similarity Explorer

• Real jewelry images  
• CLIP embeddings + cosine similarity  
• PCA projection with spherical normalization  
• Hover for metadata, side panel for preview  
""")

col1, col2 = st.columns([3, 1])

if "last_event" not in st.session_state:
    st.session_state.last_event = None

# ---------- LEFT: Visible Plot ----------
with col1:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": True}
    )

    # Invisible hover listener (DO NOT REMOVE)
    events = plotly_events(
        fig,
        hover_event=True,
        click_event=False,
        select_event=False,
        key="plotly_hover_listener",
        override_height=10,
    )

    if events:
        st.session_state.last_event = events[0]

# ---------- RIGHT: Image Preview ----------
with col2:
    st.subheader("Image Preview")

    e = st.session_state.last_event
    if e is None:
        st.info("Hover over a point to preview the image.")
    else:
        curve = e["curveNumber"]
        idx = e["pointIndex"]

        if curve == 0:
            st.image(data_image_paths[idx], use_column_width=True)
            st.markdown(
                f"""
                **Type:** Jewelry Image  
                **Index:** {idx}  
                **Closest Anchor:** A{closest_anchor_idx[idx]}  
                **Cosine Similarity:** `{max_similarity[idx]:.3f}`
                """
            )
        else:
            st.image(anchor_image_paths[idx], use_column_width=True)
            st.markdown(
                f"""
                **Type:** Anchor Image  
                **Anchor:** A{idx}
                """
            )
