import streamlit as st
import cv2
import numpy as np
import os


# Titre de l'application
st.title("Identification d'individus par l'iris")

# Ajouter un champ pour charger l'image requête
uploaded_file = st.file_uploader("Choisir une image requête...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Charger l'image requête et l'afficher
    image_requete = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image_requete, caption="Image requête", use_column_width=True)

    # Chemin vers le dossier contenant les images prétraitées
    dossier_images_pretraitees = "/content/drive/MyDrive/imagepretraites"

    # Initialiser le détecteur SIFT
    sift = cv2.SIFT_create()

    # Liste pour stocker les descripteurs SIFT de toutes les images
    descripteurs_toutes_images = []

    # Itérer sur toutes les images prétraitées dans le dossier
    for nom_fichier in os.listdir(dossier_images_pretraitees):
        chemin_image_pretraitee = os.path.join(dossier_images_pretraitees, nom_fichier)
        # Charger l'image prétraitée
        image_pretraitee = cv2.imread(chemin_image_pretraitee, cv2.IMREAD_GRAYSCALE)
        if image_pretraitee is not None:
            # Trouver les points d'intérêt et les descripteurs SIFT
            points_cles, descripteurs = sift.detectAndCompute(image_pretraitee, None)
            # Ajouter les descripteurs à la liste
            if descripteurs is not None:
                descripteurs_toutes_images.append(descripteurs)

    # Concaténer tous les descripteurs en une seule matrice
    descripteurs_toutes_images = np.concatenate(descripteurs_toutes_images, axis=0)

    # Afficher le nombre total de descripteurs extraits
    st.write(f"Nombre total de descripteurs SIFT extraits : {descripteurs_toutes_images.shape[0]}")

    # Calculer les distances euclidiennes entre le descripteur de l'image requête et chaque modèle
    distances = [np.linalg.norm(descripteurs - descripteurs_toutes_images) for descripteurs in descripteurs_toutes_images]

    # Seuil pour décider de l'acceptation ou du rejet
    threshold = 6000  # À définir selon vos besoins

    # Trouver l'indice du modèle avec la distance minimale
    min_distance_index = np.argmin(distances)

    # Vérifier si la distance minimale est inférieure au seuil donné
    if distances[min_distance_index] < threshold:
        # La correspondance est réussie, la personne est identifiée avec succès
        st.success("La personne a été identifiée avec succès !")
        st.write("Indice du modèle correspondant :", min_distance_index)
    else:
        # La correspondance a échoué, la personne n'est pas identifiée
        st.error("La personne n'a pas été identifiée.")

