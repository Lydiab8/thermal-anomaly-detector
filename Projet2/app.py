
# =============================
# Projet : Générateur d'images à partir de texte
# Auteur : Lydia
# =============================

import torch
from diffusers import StableDiffusionPipeline  # Pipeline pour générer des images
from PIL import Image                           # Manipulation d'images
import gradio as gr                             # Interface web

# -----------------------------
# 1) Choix du modèle Stable Diffusion
# -----------------------------
# Ici on utilise un modèle public déjà pré-entraîné
model_id = "runwayml/stable-diffusion-v1-5"

# Détection automatique du GPU si disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement du pipeline depuis Hugging Face
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
pipe = pipe.to(device)  # envoyer le modèle sur le GPU (ou CPU)

# -----------------------------
# 2) Fonction pour générer une image
# -----------------------------
def generate(prompt, style="None", steps=25, guidance=7.5):
    """
    prompt : texte décrivant l'image à générer
    style : style artistique ("Van Gogh", "Cartoon", "Manga", etc.)
    steps : nombre d'étapes pour la génération (qualité)
    guidance : combien le modèle suit le prompt (plus élevé = plus précis)
    """
    # Ajouter le style au prompt si fourni
    if style != "None":
        prompt = f"{prompt}, in {style} style, high quality, detailed"

    # Génération de l'image
    result = pipe(
        prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance)
    )

    # Diffusers renvoie une liste de PIL Images
    img = result.images[0]

    # Conversion et redimensionnement pour cohérence
    img = img.convert("RGB")
    img = img.resize((512, 512), resample=Image.LANCZOS)

    return img

# -----------------------------
# 3) Interface Gradio
# -----------------------------
# On crée une interface web simple
iface = gr.Interface(
    fn=generate,  # fonction appelée pour générer l'image
    inputs=[
        gr.Textbox(lines=2, placeholder="Décris ton image ici...", label="Prompt"),
        gr.Dropdown(choices=["None", "Van Gogh", "Cartoon", "Manga", "Cyberpunk"], 
                    value="None", label="Style"),
        gr.Slider(10, 50, value=25, step=1, label="Steps"),
        gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="Guidance")
    ],
    outputs=gr.Image(type="pil"),
    title="Générateur d'images avec contrôle de style",
    description="Entrez un texte, choisissez un style et cliquez sur 'Submit' pour générer une image."
)

# Lancer l'application
if __name__ == "__main__":
    iface.launch()
