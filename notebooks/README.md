# 🩺 Détection de Pneumonie par Radiographie Thoracique

> Classification binaire (NORMAL / PNEUMONIA) à partir d'images de radiographies pulmonaires, via deux approches Deep Learning : un CNN personnalisé et le Transfer Learning avec DenseNet121.

---

## 📋 Table des matières

- [Aperçu du projet](#aperçu-du-projet)
- [Dataset](#dataset)
- [Structure du projet](#structure-du-projet)
- [Modèles](#modèles)
- [Résultats](#résultats)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Technologies utilisées](#technologies-utilisées)

---

## Aperçu du projet

Ce projet vise à détecter automatiquement la **pneumonie** à partir de radiographies thoraciques (chest X-rays) en utilisant des techniques de Deep Learning. Deux architectures ont été développées et comparées :

1. **CNN personnalisé** — architecture construite from scratch avec des blocs convolutifs.
2. **DenseNet121** — modèle pré-entraîné sur ImageNet, adapté via Transfer Learning et Fine Tuning.

L'objectif médical est d'obtenir un **Recall élevé** sur la classe PNEUMONIA afin de minimiser les faux négatifs (cas de pneumonie non détectés), ce qui est critique dans un contexte clinique.

---

## Dataset

Le dataset utilisé est le **Chest X-Ray Images (Pneumonia)** disponible sur Kaggle.

| Split      | NORMAL | PNEUMONIA | Total  |
| ---------- | ------ | --------- | ------ |
| Train      | ~1 341 | ~3 875    | ~5 216 |
| Validation | subset | subset    | 20%    |
| Test       | 234    | 390       | 624    |

**Classes :**

- `NORMAL` — poumons sains
- `PNEUMONIA` — poumons infectés (bactérienne ou virale)

**Pré-traitement :**

- Redimensionnement des images : **224 × 224 px**
- Normalisation des pixels : `[0, 1]`
- Split validation : **20%** du training set (`validation_split=0.2`)

---

## Structure du projet

```
📦 pneumonia-detection/
├── app.py
├── 📓 CNN_Pneumonia.ipynb           # Notebook CNN personnalisé
├── 📓 DenseNet121_Pneumonia.ipynb   # Notebook Transfer Learning DenseNet121
├── 🧠 pneumonia_cnn_model.keras     # Modèle CNN sauvegardé
├── 🧠 densenet121_model.keras       # Modèle DenseNet121 sauvegardé
├── 📄 requirements.txt              # Dépendances Python
└── 📄 README.md                     # Documentation du projet
```

---

## Modèles

### 1. CNN Personnalisé (`CNN_Pneumonia.ipynb`)

Architecture construite from scratch avec les composants suivants :

- **Input** : `(224, 224, 3)`
- **Data Augmentation** : RandomFlip, RandomRotation (0.02), RandomZoom (0.1), RandomTranslation
- **Blocs CNN** : plusieurs couches `Conv2D` + `MaxPooling2D` + `Dropout`
- **Tête de classification** : `GlobalAveragePooling2D` → `Dense` → sortie sigmoïde
- **Optimizer** : Adam (`lr = 0.0001`)
- **Loss** : Binary Crossentropy
- **Métriques** : Accuracy, Precision, Recall, AUC
- **Callbacks** : EarlyStopping (patience=5), ModelCheckpoint
- **Epochs max** : 25 | **Batch size** : 32
- **Threshold optimal** : **0.7**

---

### 2. DenseNet121 — Transfer Learning (`DenseNet121_Pneumonia.ipynb`)

Approche en deux phases :

**Phase 1 — Feature Extraction**

- Base DenseNet121 pré-entraînée sur ImageNet, couches gelées (`trainable=False`)
- Tête : `GlobalAveragePooling2D` → `Dense` → `Dropout` → sortie sigmoïde
- Optimizer : Adam (`lr = 0.0001`)
- Epochs : 15

**Phase 2 — Fine Tuning**

- Dégel des dernières couches de la base DenseNet121
- Réentraînement avec un learning rate réduit
- Epochs : 10

**Data Augmentation** : RandomFlip, RandomRotation (0.05), RandomZoom, RandomContrast

---

## Résultats

### CNN Personnalisé — Test Set (threshold = 0.7)

| Métrique     | NORMAL | PNEUMONIA | Global     |
| ------------ | ------ | --------- | ---------- |
| Precision    | 0.72   | 0.88      | 0.82       |
| Recall       | 0.81   | 0.81      | 0.81       |
| F1-score     | 0.76   | 0.84      | 0.81       |
| **Accuracy** |        |           | **81%**    |
| **F1 macro** |        |           | **0.8415** |

---

### DenseNet121 — Test Set (threshold = 0.8, après Fine Tuning)

| Métrique     | NORMAL | PNEUMONIA | Global     |
| ------------ | ------ | --------- | ---------- |
| Precision    | 0.80   | 0.93      | 0.89       |
| Recall       | 0.90   | 0.87      | 0.88       |
| F1-score     | 0.85   | 0.90      | 0.88       |
| **Accuracy** |        |           | **88%**    |
| **F1 macro** |        |           | **0.9109** |

---

### Comparaison des deux modèles

| Modèle      | Accuracy  | F1 (PNEUMONIA) | Recall (PNEUMONIA) | AUC       |
| ----------- | --------- | -------------- | ------------------ | --------- |
| CNN custom  | 80.3%     | 0.84           | 0.81               | ~0.89     |
| DenseNet121 | **88.9%** | **0.91**       | **0.92**           | **~0.95** |

> **DenseNet121 surpasse le CNN personnalisé** sur toutes les métriques grâce au Transfer Learning depuis ImageNet.

---

## Installation

### Prérequis

- Python 3.8+
- GPU recommandé (entraînement sur Google Colab avec GPU)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

**`requirements.txt` :**

```
streamlit
tensorflow
numpy
matplotlib
plotly
pillow
```

### Dataset

Télécharger le dataset depuis Kaggle :

```
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```

Extraire dans `/content/chest_xray/chest_xray/` (structure attendue) :

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Utilisation

### Entraînement

Ouvrir et exécuter les notebooks dans Google Colab (GPU recommandé) :

```
CNN_Pneumonia.ipynb        → CNN from scratch
DenseNet121_Pneumonia.ipynb → Transfer Learning DenseNet121
```

### Chargement d'un modèle sauvegardé

```python
from tensorflow.keras.models import load_model

# CNN
model = load_model("pneumonia_cnn_model.keras")

# DenseNet121
model = load_model("densenet121_model.keras")
```

### Prédiction sur une image

```python
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

def predict_image(model, image_path, threshold=0.7):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction >= threshold else "NORMAL"

    return label, float(prediction)

label, score = predict_image(model, "chest_xray.jpg")
print(f"Diagnostic : {label} (score : {score:.4f})")
```

---

## Technologies utilisées

| Technologie        | Usage                                                   |
| ------------------ | ------------------------------------------------------- |
| TensorFlow / Keras | Construction et entraînement des modèles                |
| DenseNet121        | Transfer Learning (ImageNet)                            |
| NumPy              | Manipulation des arrays                                 |
| Matplotlib         | Visualisation des courbes d'entraînement                |
| Plotly             | Graphiques interactifs                                  |
| Pillow (PIL)       | Chargement et traitement des images                     |
| Streamlit          | Interface de démo (optionnelle)                         |
| Scikit-learn       | Métriques (classification report, matrice de confusion) |
| Google Colab       | Environnement d'entraînement (GPU)                      |

---

## Remarques

- Le **seuil de décision (threshold)** a été optimisé empiriquement pour chaque modèle afin de maximiser le F1-score sur la classe PNEUMONIA.
- L'utilisation de **Data Augmentation** (flip, rotation, zoom, translation/contraste) améliore la généralisation sur des données médicales limitées.
- Dans un contexte médical réel, il est recommandé de **privilégier le Recall** sur la Precision pour la classe PNEUMONIA afin de minimiser les faux négatifs.

---

_Projet réalisé dans le cadre d'une étude comparative des architectures Deep Learning pour la classification d'images médicales._
