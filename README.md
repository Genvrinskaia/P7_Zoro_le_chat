# 🐱 Cat Keypoints Detection App

Application Gradio pour prédire la position de la tête des chats à partir d'images, en utilisant un modèle **Vision Transformer (ViT)** pré-entraîné et fine-tuné.

Le modèle prédit **9 keypoints** (2 yeux, 6 pour les oreilles, 1 pour la bouche) sur des images de chats.

---

## Fichiers inclus

- `app.py` : application Gradio (prédiction, EDA, accessibilité WCAG)
- `model.py` : architecture `ViTForKeypointRegression`
- `requirements.txt` : dépendances Python
- `README.md` : ce fichier

---

## Modèle pré-entraîné

Le modèle `vit_lr_1e-5.pth` est téléchargé automatiquement et gratuitement depuis [Hugging Face](https://huggingface.co/Genvrin/vit_cat_keypoints).

---

## Lancer l'application en local

```bash
pip install -r requirements.txt
python app.py
