import streamlit as st
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from datetime import datetime
import io

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer les classes des modèles (suppression du modèle de diffusion)
from src.models.transformer.transformer_model import TransformerModel, clean_repetitions
from src.models.cnn.deep_cnn import DeepCNN
from src.creative.latent_exploration import LatentExplorer

# Configuration de la page
st.set_page_config(
    page_title="Modèles Génératifs - Visualisation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .model-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #ccc;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .output-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .hint-text {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Variables globales pour stocker les modèles chargés
@st.cache_resource
def load_transformer_model(model_path):
    """Charge le modèle Transformer avec mise en cache"""
    try:
        model = TransformerModel(
            vocab_size=50257,
            d_model=128,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        return model, True
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle Transformer: {e}")
        return None, False

@st.cache_resource
def load_cnn_model(model_path):
    """Charge le modèle CNN/GAN avec mise en cache"""
    try:
        model = DeepCNN(input_shape=(64, 64, 3), latent_dim=100)
        model.load_model(model_path)
        return model, True
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle GAN: {e}")
        return None, False

def find_latest_model(model_type):
    """Trouve le chemin du dernier modèle entraîné d'un type spécifique"""
    if model_type == "transformer":
        pattern = "results/transformer_*/models/transformer_final.pt"
    elif model_type == "cnn":
        pattern = "checkpoints/*/gan_checkpoint_*_generator.h5"
    else:
        return None
    
    paths = glob.glob(pattern)
    if not paths:
        return None
    
    return max(paths, key=os.path.getctime)

def create_sidebar():
    """Crée la barre latérale avec les options de chargement des modèles"""
    st.sidebar.markdown("## Configuration des Modèles")
    
    # Transformer
    st.sidebar.markdown("### Transformer (Texte)")
    transformer_use_latest = st.sidebar.checkbox("Utiliser le dernier modèle Transformer", value=True)
    transformer_path = find_latest_model("transformer")
    
    if transformer_use_latest:
        st.sidebar.text(f"Modèle: {transformer_path}")
    else:
        transformer_path = st.sidebar.text_input("Chemin du modèle Transformer", value=transformer_path or "")
    
    # CNN/GAN
    st.sidebar.markdown("### GAN (Images)")
    cnn_use_latest = st.sidebar.checkbox("Utiliser le dernier modèle GAN", value=True)
    cnn_path = find_latest_model("cnn")
    
    if cnn_use_latest:
        st.sidebar.text(f"Modèle: {cnn_path}")
    else:
        cnn_path = st.sidebar.text_input("Chemin du modèle GAN", value=cnn_path or "")
    
    return transformer_path, cnn_path

def transformer_ui(model_path):
    """Interface utilisateur pour le modèle Transformer"""
    st.markdown('<div class="model-header">Transformer - Génération de Texte</div>', unsafe_allow_html=True)
    
    if not model_path:
        st.warning("Aucun modèle Transformer trouvé. Veuillez spécifier un chemin dans la barre latérale.")
        return
    
    # Charger le modèle si nécessaire
    model, success = load_transformer_model(model_path)
    if not success:
        return
    
    # Interface utilisateur
    prompt = st.text_input("Prompt de départ:", value="Once upon a time")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider("Température:", min_value=0.1, max_value=2.0, value=0.7, step=0.1, 
                              help="Contrôle la créativité (plus élevé = plus créatif)")
    with col2:
        top_k = st.slider("Top-K:", min_value=1, max_value=100, value=50, 
                        help="Limite le choix aux K tokens les plus probables")
    with col3:
        top_p = st.slider("Top-P:", min_value=0.1, max_value=1.0, value=0.95, step=0.05, 
                        help="Limite aux tokens couvrant P% de la probabilité cumulative")
    
    col1, col2 = st.columns(2)
    with col1:
        repetition_penalty = st.slider("Pénalité de répétition:", min_value=1.0, max_value=3.0, value=2.0, step=0.1,
                                    help="Pénalise les tokens répétés (plus élevé = moins de répétitions)")
    with col2:
        max_length = st.slider("Longueur maximale:", min_value=50, max_value=500, value=200, step=10)
    
    # Génération de texte
    if st.button("Générer du texte"):
        with st.spinner("Génération en cours..."):
            generated_text = model.generate(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            # Post-traitement pour améliorer la qualité
            cleaned_text = clean_repetitions(generated_text)
            
            # Afficher le résultat
            st.markdown('<div class="section-header">Texte généré:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{cleaned_text}</div>', unsafe_allow_html=True)
            
            # Boutons pour sauvegarder et partager
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Sauvegarder comme .txt"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"transformer_generation_{timestamp}.txt"
                    with open(os.path.join("results", filename), "w", encoding="utf-8") as f:
                        f.write(cleaned_text)
                    st.success(f"Sauvegardé dans results/{filename}")

def gan_ui(model_path):
    """Interface utilisateur améliorée pour le modèle GAN"""
    st.markdown('<div class="model-header">GAN - Génération d\'Images</div>', unsafe_allow_html=True)
    
    if not model_path:
        st.warning("Aucun modèle GAN trouvé. Veuillez spécifier un chemin dans la barre latérale.")
        return
    
    # Charger le modèle si nécessaire
    model, success = load_cnn_model(model_path)
    if not success:
        return
    
    # Interface améliorée avec tabs
    tabs = st.tabs(["Génération basique", "Paramètres avancés", "Exploration latente"])
    
    with tabs[0]:
        st.markdown('<div class="section-header">Génération d\'images</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            num_images = st.slider("Nombre d'images:", min_value=1, max_value=16, value=4)
        with col2:
            image_size = st.selectbox("Taille des images:", options=["64x64", "128x128"], index=0)
        
        # Génération d'images
        if st.button("Générer des images"):
            with st.spinner("Génération en cours..."):
                try:
                    generated_images = model.generate_images(num_images=num_images)
                    
                    # Afficher les images générées
                    st.markdown('<div class="section-header">Images générées:</div>', unsafe_allow_html=True)
                    
                    # Organiser en grille
                    cols = min(4, num_images)
                    rows = (num_images + cols - 1) // cols
                    
                    for i in range(rows):
                        row_cols = st.columns(cols)
                        for j in range(cols):
                            idx = i * cols + j
                            if idx < num_images:
                                img = generated_images[idx]
                                # Normaliser si nécessaire
                                if img.min() < 0:
                                    img = (img + 1) / 2
                                if isinstance(img, torch.Tensor):
                                    img = img.detach().cpu().numpy().transpose(1, 2, 0)
                                elif isinstance(img, np.ndarray) and img.shape[0] == 3:
                                    img = img.transpose(1, 2, 0)
                                row_cols[j].image(img, caption=f"Image {idx+1}", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la génération d'images: {e}")
    
    with tabs[1]:
        st.markdown('<div class="section-header">Paramètres avancés de génération</div>', unsafe_allow_html=True)
        
        truncation = st.slider("Troncature:", min_value=0.1, max_value=1.5, value=0.8, step=0.1,
                             help="Valeurs plus basses donnent des images plus 'moyennes', plus élevées plus diversifiées")
        
        # Seed for reproducibility
        use_seed = st.checkbox("Utiliser un seed (reproductibilité)", value=False)
        seed = st.number_input("Seed:", value=42, disabled=not use_seed)
        
        if st.button("Générer avec paramètres avancés"):
            with st.spinner("Génération en cours..."):
                try:
                    # Set seed if needed
                    if use_seed:
                        np.random.seed(int(seed))
                    
                    # Generate images with truncation
                    generated_images = model.generate_images(num_images=4, truncation=truncation)
                    
                    # Display images in 2x2 grid
                    st.markdown('<div class="section-header">Images générées avec paramètres avancés:</div>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    col3, col4 = st.columns(2)
                    
                    cols = [col1, col2, col3, col4]
                    for i, img in enumerate(generated_images[:4]):
                        if img.min() < 0:
                            img = (img + 1) / 2
                        cols[i].image(img, caption=f"Image {i+1} (truncation={truncation})", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la génération d'images: {e}")
    
    with tabs[2]:
        st.markdown('<div class="section-header">Exploration latente avancée</div>', unsafe_allow_html=True)
        
        exploration_type = st.selectbox(
            "Type d'exploration:",
            options=["Interpolation linéaire", "Mélange de styles", "Marche aléatoire"],
            index=0
        )
        
        if exploration_type == "Interpolation linéaire":
            steps = st.slider("Nombre d'étapes d'interpolation:", min_value=5, max_value=30, value=10)
            
            if st.button("Générer l'interpolation"):
                with st.spinner("Création de l'interpolation..."):
                    try:
                        explorer = LatentExplorer(model)
                        
                        # Générer les images interpolées
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join("results", f"latent_interpolation_{timestamp}.gif")
                        explorer.latent_space_interpolation(n_steps=steps, animate=True, save_path=output_path)
                        
                        # Afficher le GIF
                        with open(output_path, 'rb') as f:
                            gif_data = f.read()
                            st.image(gif_data, caption="Interpolation dans l'espace latent")
                        
                    except Exception as e:
                        st.error(f"Erreur lors de l'exploration latente: {e}")
                        
        elif exploration_type == "Mélange de styles":
            num_sources = st.slider("Nombre d'images sources:", min_value=2, max_value=5, value=3)
            
            if st.button("Générer le mélange de styles"):
                with st.spinner("Création du mélange de styles..."):
                    try:
                        explorer = LatentExplorer(model)
                        
                        # Générer la grille de mélanges de style
                        style_grid = explorer.style_mixing(num_samples=num_sources)
                        
                        # Afficher la grille
                        st.image(style_grid, caption="Mélange de styles (lignes/colonnes = images sources)", use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"Erreur lors du mélange de styles: {e}")

def main():
    st.markdown('<div class="main-header">Visualisation des Modèles Génératifs</div>', unsafe_allow_html=True)
    
    st.write("""
    Cette application permet de visualiser et d'interagir avec deux types de modèles génératifs développés dans ce projet:
    - **Transformer**: Génération de texte avec contrôle de divers paramètres
    - **GAN (CNN)**: Génération d'images et exploration de l'espace latent
    
    Utilisez les contrôles dans la barre latérale pour configurer les modèles, puis interagissez avec chaque section ci-dessous.
    """)
    
    # Configuration des modèles via la barre latérale (suppression du modèle de diffusion)
    transformer_path, cnn_path = create_sidebar()
    
    # Interface pour chaque modèle (suppression du modèle de diffusion)
    transformer_ui(transformer_path)
    gan_ui(cnn_path)
    
    # Footer
    st.markdown("---")
    st.markdown('<div class="hint-text">Projet de modèles génératifs - Technologies utilisées: PyTorch, TensorFlow, Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()