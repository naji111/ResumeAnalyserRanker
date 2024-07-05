# Resume Analyser Ranker 📄🔍📈

## Table des matières
1. [Introduction](#introduction)
2. [Fonctionnalités](#fonctionnalités)
3. [Prérequis](#prérequis)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Utilisation](#utilisation)
7. [Structure du projet](#structure-du-projet)
8. [Détails techniques](#détails-techniques)
9. [Contribution](#contribution)
10. [Contact](#contact)
11. [Technologies](#technologies)

## Introduction 🚀

Resume Analyser Ranker est une application web Flask conçue pour automatiser et optimiser le processus de sélection des CV. En utilisant des techniques avancées de traitement du langage naturel (NLP) et d'apprentissage automatique, ce projet analyse, évalue et classe les CV en fonction de descriptions de poste spécifiques.

L'objectif principal est de simplifier et d'accélérer le processus de recrutement en fournissant une analyse approfondie et un classement objectif des candidats potentiels. Cela permet aux recruteurs et aux responsables RH de gagner un temps précieux et de prendre des décisions plus éclairées lors de la présélection des candidats.

## Fonctionnalités ⚙️

- Téléchargement et analyse simultanés de plusieurs CV et descriptions de poste
- Extraction automatique d'informations clés des CV :
  - Nom du candidat
  - Numéro de téléphone
  - Adresse e-mail
  - Compétences
  - Années d'expérience
  - Localisation
  - Noms des entreprises précédentes
- Calcul de la similarité entre les CV et les descriptions de poste
- Classement des CV les plus pertinents pour chaque poste
- Génération de nuages de mots pour visualiser les termes clés des CV
- Interface utilisateur intuitive pour la gestion des utilisateurs et l'affichage des résultats
- Stockage sécurisé des données utilisateur et des résultats d'analyse dans MongoDB

## Prérequis 📋

- Python 3.8+
- MongoDB
- Pip (gestionnaire de paquets Python)
- Git

## Installation 💻

1. Clonez le dépôt :

```
git clone https://github.com/naji111/Resume_Analyser_Ranker.git cd Resume_Analyser_Ranker
```

3. Créez un environnement virtuel :

Sur Mac : 
```
python -m venv venv source venv/bin/activate
```
Sur Windows : 
```
venv\Scripts\activate
```

5. Installez les dépendances :

```
pip install -r requirements.txt
```

7. Téléchargez les modèles de langue nécessaires :

```
python -m spacy download fr_core_news_sm python -m nltk.downloader punkt maxent_ne_chunker words averaged_perceptron_tagger wordnet stopwords
```


## Configuration ⚙️

1. Assurez-vous que MongoDB est installé et en cours d'exécution sur votre système.

2. Créez un fichier `.env` à la racine du projet et ajoutez les variables d'environnement suivantes :

```
MONGODB_URI=mongodb://localhost:27017/ SECRET_KEY=votre_clé_secrète_ici
```

3. Modifiez le chemin du fichier Skills.xlsx dans le code si nécessaire.

## Utilisation 🎮

1. Lancez l'application :
python app.py


2. Ouvrez un navigateur et accédez à [link](http://localhost:5000).

3. Créez un compte utilisateur ou connectez-vous.

4. Téléchargez les CV et les descriptions de poste.

5. Lancez l'analyse et visualisez les résultats.

## Structure du projet 🏗️

```
Resume_Analyser_Ranker/
├── .env             # Fichier de configuration (à créer) 
├── app.py           # Point d'entrée principal de l'application Flask 
├── data/ 
│ └── Skills.xlsx    # Liste des compétences prédéfinies 
├── static/
│ ├── css
│ └── wordcloud.png  # Image du nuage de mots généré 
├── templates/  
│ ├── index.html     # Page d'accueil 
│ ├── signup.html    # Page d'inscription 
│ ├── login.html     # Page de connexion 
│ ├── upload.html    # Page de téléchargement des fichiers 
│ └── results.html   # Page d'affichage des résultats
├── requirements.txt # Liste des dépendances Python 
└── README.md        # Documentation du projet
```

## Détails techniques 🧪

### Traitement des CV

1. **Extraction de texte** : Utilisation de `pdfminer3` pour les PDF et `mammoth` pour les fichiers DOCX.
2. **Prétraitement** : Suppression des mots vides, des signes de ponctuation, des caractères spéciaux et normalisation du texte.
3. **Extraction d'informations** : Utilisation d'expressions régulières et de techniques NLP pour extraire les informations clés.
4. **Analyse des compétences** : Comparaison avec une liste prédéfinie de compétences.
5. **Calcul de similarité** : Utilisation de TF-IDF et de la similarité cosinus pour comparer les CV aux descriptions de poste.

### Sécurité

- Hachage des mots de passe avec SHA-256.
- Utilisation de sessions Flask pour l'authentification des utilisateurs.
- Stockage sécurisé des données dans MongoDB.

### Visualisation

- Génération de nuages de mots avec la bibliothèque `wordcloud`.

### Contribution 🤝

Les contributions sont les bienvenues ! Pour contribuer :
- Forkez le projet
- Créez votre branche de fonctionnalité (git checkout -b feature/AmazingFeature)
- Committez vos changements (git commit -m 'Add some AmazingFeature')
- Poussez vers la branche (git push origin feature/AmazingFeature)
- Ouvrez une Pull Request

### Contact 📞
Naji 
- @naji111
- najiezzoubir23@gmail.com
- [Lien du projet : Resume Analyser Ranker](https://github.com/naji111/Resume_Analyser_Ranker)

### Technologies 💻
#### Flask
<div>
  <img src="https://tse3.mm.bing.net/th?id=OIP.tZKxFU0lwHLBBNMxk53WfAHaJh&pid=Api&P=0&h=180" width=200px>
</div>

#### MongoDB
<div>
  <img src="https://1000logos.net/wp-content/uploads/2020/08/MongoDB-Logo-1536x960.png" width=200px>
</div>

#### NLTK
<div>
  <img src="https://tse2.mm.bing.net/th?id=OIP.UC9wrYxc1EWJlB0lkEQyYwHaID&pid=Api&P=0&h=180" width=200px>
</div>

#### spaCy
<div>
  <img src="https://uploads-ssl.webflow.com/5fdc17d51dc102ed1cf87c05/603e718b9010e3c001777a25_spacy.png" width=200px>
</div>

#### scikit-learn
<div>
  <img src="https://logosdownload.com/logo/scikit-learn-logo-big.png" width=200px>
</div>

#### WordCloud
<div>
  <img src="https://i.pinimg.com/originals/29/7b/de/297bde9d86a727b0f0d4b5b683dad490.png" width=200px>
</div>

#### pdfminer3 
<div>
  <img src="https://2.bp.blogspot.com/-X_KYPdTzq0g/WZ0L0IEiXjI/AAAAAAAAKlI/EkSOM7pb2mIX6CyfAda-exWIpKI-gDYoACLcBGAs/s1600/PDFMiner.jpg" width=200px>
</div>

