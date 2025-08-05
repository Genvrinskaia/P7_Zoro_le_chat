# app.py

# ⚠️ Ce script prend en compte certains critères WCAG pour améliorer l'accessibilité visuelle
# entre autre, des émojis pour améliorer l'accéssibilité cognitive, mais pas que ça.

import os
import requests
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import gradio as gr
from torchvision import transforms
from model import ViTForKeypointRegression

# --- Téléchargement automatique du modèle depuis Hugging Face ---
MODEL_URL = "https://huggingface.co/Genvrin/vit_cat_keypoints/resolve/main/vit_lr_1e-5.pth"
MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "vit_lr_1e-5.pth")

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Modèle téléchargé.")
    else:
        print("Modèle déjà présent.")

download_model()

# --- Chemins ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data_sample")

# --- Vérifications ---
print(f"[INFO] DATA_PATH = {DATA_PATH}")
print(f"[INFO] Contenu de data/ : {os.listdir(DATA_PATH)}")

CATEGORIES = sorted([
    f for f in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, f))
])

print(f"[INFO] CATEGORIES = {CATEGORIES}")

# --- Constantes ---
NUM_KEYPOINTS = 9

# --- Prétraitement ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# --- Chargement du modèle ---
model = ViTForKeypointRegression(num_keypoints=NUM_KEYPOINTS)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# --- Prédiction ---
def predict(img: Image.Image):
    image = img.convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor).squeeze(0)

    keypoints_norm = torch.sigmoid(outputs).view(NUM_KEYPOINTS, 2).numpy()
    orig_w, orig_h = image.size
    keypoints = keypoints_norm.copy()
    keypoints[:, 0] *= orig_w
    keypoints[:, 1] *= orig_h

    fig, ax = plt.subplots()
    ax.imshow(image)

    x_min, y_min = keypoints.min(axis=0)
    x_max, y_max = keypoints.max(axis=0)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    radius = np.sqrt(((keypoints - [center_x, center_y]) ** 2).sum(axis=1)).max()

    # ✅ Couleurs contrastées (WCAG 1.4.1)
    circle = Circle((center_x, center_y), radius, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(circle)
    ax.axis("off")

    return fig

# --- Exploratoire ---
def get_images_by_category(category):
    folder = os.path.join(DATA_PATH, category)
    image_paths = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
    images = []
    for filename in image_paths[:6]:
        path = os.path.join(folder, filename)
        images.append(Image.open(path).convert("RGB"))
    return images

def plot_category_counts(selected_cat=None):
    counts = []
    labels = []
    colors = []
    for cat in CATEGORIES:
        folder = os.path.join(DATA_PATH, cat)
        jpg_files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
        labels.append(cat)
        counts.append(len(jpg_files))
        colors.append("#8BC34A" if selected_cat == cat else "lightgray")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black")
    for bar, label, color in zip(bars, labels, colors):
        if color == "#8BC34A":
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, label,
                    ha='center', fontsize=10, fontweight='bold', color='#388E3C')
    ax.set_title("Distribution des images dans les dossiers CAT_00 à CAT_06", fontsize=14)
    ax.set_xlabel("Numéro du dossier")
    ax.set_ylabel("Nombre d'images")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    return fig

def plot_rgb_histograms(category):
    folder = os.path.join(DATA_PATH, category)
    image_paths = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])[:6]
    fig, axes = plt.subplots(len(image_paths), 1, figsize=(10, 3 * len(image_paths)))
    if len(image_paths) == 1:
        axes = [axes]
    for i, filename in enumerate(image_paths):
        path = os.path.join(folder, filename)
        image = Image.open(path).convert("RGB")
        img_array = np.array(image)

        ax_img = fig.add_axes([0.05, 1 - (i + 1) * (1 / len(image_paths)), 0.1, 0.1])
        ax_img.imshow(image.resize((50, 50)))
        ax_img.axis("off")

        ax_hist = fig.add_axes([0.2, 1 - (i + 1) * (1 / len(image_paths)), 0.75, 0.1])
        ax_hist.hist(img_array[:, :, 0].ravel(), bins=256, color='r', alpha=0.5, label='R')
        ax_hist.hist(img_array[:, :, 1].ravel(), bins=256, color='g', alpha=0.5, label='G')
        ax_hist.hist(img_array[:, :, 2].ravel(), bins=256, color='b', alpha=0.5, label='B')
        ax_hist.set_xlim([0, 256])
        ax_hist.legend()
        ax_hist.set_title(f"Image : {filename}", fontsize=10)
    fig.tight_layout()
    return fig

# --- Prédiction ---
def get_gallery_and_filenames(category):
    folder = os.path.join(DATA_PATH, category)
    image_paths = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
    images = [Image.open(os.path.join(folder, f)).convert("RGB") for f in image_paths[:6]]
    return images, image_paths[:6]

def predict_from_filename(category, filename):
    path = os.path.join(DATA_PATH, category, filename)
    image = Image.open(path).convert("RGB")
    return predict(image)

# --- Interface Gradio ---
with gr.Blocks(title="Dashboard Projet 7 - Keypoints de Chats") as demo:
    gr.Markdown("# 🐱 Dashboard Projet 7 - Keypoints de Chats")
    gr.Markdown("**Analyse exploratoire des données** et **prédiction de la tête** d’un chat à partir d’une image.")

    with gr.Tab("📊 Analyse Exploratoire"):
        gr.Markdown("### 🖼️ Galerie d’exemples par catégorie")
        dropdown = gr.Dropdown(choices=CATEGORIES, label="Choisir une catégorie (CAT_00 à CAT_06)")
        gallery = gr.Gallery(label="Images", columns=3, rows=2)
        dropdown.change(fn=get_images_by_category, inputs=dropdown, outputs=gallery)

        gr.Markdown("### 📊 Répartition des images par dossier")
        histogram_plot = gr.Plot(label="Histogramme des images")
        dropdown.change(fn=plot_category_counts, inputs=dropdown, outputs=histogram_plot)

        gr.Markdown("### 🌈 Histogrammes RGB des images sélectionnées")
        rgb_plot = gr.Plot(label="Histogrammes RGB")
        dropdown.change(fn=plot_rgb_histograms, inputs=dropdown, outputs=rgb_plot)

    with gr.Tab("📌 Prédiction Zone des Keypoints"):
        gr.Markdown("### 🎯 Prédiction de la tête d’un chat à partir d’une image")

        with gr.Tabs():
            with gr.Tab("🗂️ Depuis les dossiers"):
                category_dropdown = gr.Dropdown(choices=CATEGORIES, label="Choisir un dossier (CAT_00 à CAT_06)")
                project_gallery = gr.Gallery(label="Images du dossier", columns=3, rows=2)
                image_dropdown = gr.Dropdown(choices=[], label="Choisir une image")
                image_output = gr.Plot(label="Prédiction de la tête")
                predict_btn = gr.Button("Prédire les keypoints")

                def update_gallery_and_dropdown(cat):
                    images, filenames = get_gallery_and_filenames(cat)
                    return images, gr.update(choices=filenames)

                category_dropdown.change(fn=update_gallery_and_dropdown, inputs=category_dropdown, outputs=[project_gallery, image_dropdown])
                predict_btn.click(fn=predict_from_filename, inputs=[category_dropdown, image_dropdown], outputs=image_output)

            with gr.Tab("📷 Image importée"):
                input_image = gr.Image(type="pil", label="Image importée")
                output_figure = gr.Plot(label="Résultat de la prédiction")
                predict_upload_btn = gr.Button("Prédire sur image importée")

                predict_upload_btn.click(fn=predict, inputs=input_image, outputs=output_figure)

# --- Lancement adapté à Render ---
if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=8080)
