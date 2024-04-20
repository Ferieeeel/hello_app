import cv2
import numpy as np

# Extraction des descripteurs de la base de données
database_descriptors = []
for image_file in database_images:
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    database_descriptors.append(descriptors)

# Extraction des descripteurs de l'image requête
image_requete = cv2.imread(image_requete_path, cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
keypoints_requete, descriptors_requete = sift.detectAndCompute(image_requete, None)

# Comparaison des descripteurs
distances = []
for descriptor_db in database_descriptors:
    for descriptor in descriptor_db:
        distance = np.linalg.norm(descriptor - descriptors_requete)
        distances.append(distance)

# Identification de la correspondance la plus proche
min_distance_index = np.argmin(distances)

# Affichage du résultat
result_image = database_images[min_distance_index]
cv2.imshow('Correspondance la plus proche', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


