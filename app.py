import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from PIL import Image

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

@st.cache_resource
def load_model():
    model = EmbeddingNet(embedding_dim=2)
    model.load_state_dict(torch.load("contrastive_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def load_embeddings():
    data = np.load("embeddings_after_training.npz")
    if "embeddings" in data and "labels" in data:
        embeddings = data["embeddings"]
        labels = data["labels"]
    else:
        arrs = [data[k] for k in data.files]
        embeddings = arrs[0]
        labels = arrs[1] if len(arrs) > 1 else np.zeros(len(embeddings))
    return embeddings, labels

@st.cache_data
def load_cifar10_images():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return dataset

st.title("LatentLens")

model = load_model()
embeddings, labels = load_embeddings()
dataset = load_cifar10_images()

label_map = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}


st.sidebar.header("Filters")
class_names = [label_map[i] for i in sorted(set(labels))]
selected_classes = st.sidebar.multiselect("Select classes:", class_names, default=class_names)
selected_indices = [k for k, v in label_map.items() if v in selected_classes]
mask = np.isin(labels, selected_indices)
embeddings = embeddings[mask]
labels = labels[mask]


fig, ax = plt.subplots(figsize=(7, 6))
palette = sns.color_palette("tab10", len(label_map))

for idx in selected_indices:
    class_mask = labels == idx
    ax.scatter(
        embeddings[class_mask, 0],
        embeddings[class_mask, 1],
        s=10, alpha=0.7,
        label=label_map[idx],
        color=palette[idx]
    )

st.subheader(" Upload Your Own Image")

uploaded_file = st.file_uploader("Upload an image (CIFAR-like)", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Your uploaded image", width=150)

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)

    # Compute embedding
    with torch.no_grad():
        user_emb = model(img_tensor).numpy().squeeze()

    # Plot user embedding (★)
    ax.scatter(user_emb[0], user_emb[1], marker='*', s=50, color='black', label="Your Image")

    # Find 5 nearest neighbors
    dists = np.linalg.norm(embeddings - user_emb, axis=1)
    nearest_indices = np.argsort(dists)[:5]

    st.write("5 Nearest CIFAR-10 Images:")
    cols = st.columns(5)
    for i, idx in enumerate(nearest_indices):
        img_tensor, lbl = dataset[idx]
        img_np = np.transpose(img_tensor.numpy(), (1, 2, 0))
        with cols[i]:
            st.image(img_np, caption=label_map[int(lbl)], width=80)

ax.set_title("2D Embeddings After Contrastive Training", fontsize=14)
ax.legend(markerscale=2, fontsize=8, loc="best", frameon=False)
st.pyplot(fig)

