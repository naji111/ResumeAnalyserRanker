
# Importation des bibliothèques nécessaires
from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient, errors
import hashlib

import pandas as pd
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load("fr_core_news_sm")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mammoth
import locationtagger
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.corpus import wordnet
nltk.download('wordnet')
from sklearn.decomposition import TruncatedSVD
from datetime import datetime
from bson import ObjectId
from nltk import sent_tokenize





app = Flask(__name__)

app.secret_key = 'MySecretKey12345'



# Connexion MongoDB
try:
    # Essaie de se connecter à MongoDB en utilisant l'URL "mongodb://localhost:27017/"
    # avec un délai d'attente de 5 secondes pour la sélection du serveur
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    
    # Sélectionne la base de données "Resume_Analyzer_Ranker"
    db = client["Resume_Analyzer_Ranker"]
    
    # Sélectionne la collection "users" dans la base de données
    users_collection = db["users"]
    
    # Affiche un message indiquant que la connexion à MongoDB a réussi
    print("Connected to MongoDB")
    
except errors.ServerSelectionTimeoutError as err:
    # Si la connexion à MongoDB échoue en raison d'un délai d'attente dépassé,
    # affiche un message d'erreur avec le détail de l'erreur
    print(f"Failed to connect to MongoDB: {err}")






# Fonctions utilitaires pour le hachage de mot de passe
def hash_password(password):
    """
    Hache le mot de passe en utilisant l'algorithme SHA-256.
    
    Args:
        password (str): Le mot de passe à hacher.
    
    Returns:
        str: Le mot de passe haché sous forme de chaîne de caractères hexadécimale.
    """
    # Encode le mot de passe en bytes
    password_bytes = password.encode()
    
    # Hache le mot de passe en utilisant l'algorithme SHA-256
    hashed_password = hashlib.sha256(password_bytes).hexdigest()
    
    return hashed_password


def verify_password(stored_password, provided_password):
    """
    Vérifie si le mot de passe fourni correspond au mot de passe stocké.
    
    Args:
        stored_password (str): Le mot de passe stocké, haché.
        provided_password (str): Le mot de passe fourni par l'utilisateur.
    
    Returns:
        bool: True si les mots de passe correspondent, False sinon.
    """
    # Hache le mot de passe fourni
    hashed_provided_password = hash_password(provided_password)
    
    # Vérifie si le mot de passe stocké correspond au mot de passe fourni
    return stored_password == hashed_provided_password







# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Fonctions Pour Analyser CV @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




# Fonction d'extraction de PDF
def pdf_extractor(file_content):
    """
    Extrait le contenu textuel d'un fichier PDF fourni.

    Args:
        file_content (bytes): Contenu binaire brut du fichier PDF.

    Returns:
        str: Contenu textuel extrait du PDF, ou une chaîne vide si l'extraction échoue.
    """

    # Création d'un tampon de chaîne pour stocker le texte extrait
    output = io.StringIO()

    # Gestionnaire de ressources pour gérer les ressources PDF
    rsrcmgr = PDFResourceManager()

    # Convertisseur de texte en appareil pour écrire le texte extrait dans le tampon de chaîne
    device = TextConverter(rsrcmgr, output, laparams=LAParams())

    # Interpréteur de page PDF pour traiter chaque page du PDF
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    try:
        # Itérer sur toutes les pages du PDF
        for page in PDFPage.get_pages(io.BytesIO(file_content), caching=True, check_extractable=True):
            interpreter.process_page(page)  # Traiter la page actuelle

    except Exception as e:
        print(f"Erreur d'extraction du texte du PDF : {e}")
        return ""  # Retourner une chaîne vide en cas d'erreur

    # Récupérer le contenu textuel extrait du tampon de chaîne
    text = output.getvalue()

    # Fermer le tampon de chaîne (facultatif, car le gestionnaire de contexte gère la fermeture)
    output.close()

    return text




# Fonction pour lire des fichiers
def read_files(file_path):
    """
    Lit et extrait le contenu textuel d'une liste de fichiers fournis sous forme de chemins d'accès aux fichiers.

    Args:
        file_path (list): Une liste contenant les chemins d'accès aux fichiers à traiter.

    Returns:
        list: Une liste du contenu textuel extrait de chaque fichier, ou une liste vide si aucun fichier n'a été trouvé.
    """

    textes_extraits = []

    for fichier_televerse in file_path:
        try:
            # Extraire le nom du fichier et lire le contenu du fichier
            nom_fichier = fichier_televerse.filename
            contenu_fichier = fichier_televerse.read()

            # Déterminer le type de fichier en fonction de l'extension
            if nom_fichier.endswith(".pdf"):
                textes_extraits.append(pdf_extractor(contenu_fichier))
            elif nom_fichier.endswith(".docx") or nom_fichier.endswith(".doc"):
                # Gérer les formats DOCX et DOC avec textract
                texte = textract.process(contenu_fichier).decode("utf-8")
                textes_extraits.append(texte)
            elif nom_fichier.endswith(".txt"):
                # Lire le contenu textuel directement pour les fichiers texte brut
                contenu = contenu_fichier.decode("utf-8")
                textes_extraits.append(contenu)
            else:
                print(f"Format de fichier non pris en charge : {nom_fichier}")

        except Exception as e:
            print(f"Erreur lors du traitement du fichier {nom_fichier} : {e}")

    return textes_extraits




# Fonction de prétraitement
def preprocessing(Txt):
    """
    Prétraite du texte pour le nettoyage et la préparation à l'analyse.

    Args:
        Txt (list): Une liste contenant du texte à prétraiter.

    Returns:
        list: Une liste contenant le texte prétraité.
    """

    # Liste de stop words en français
    sw = stopwords.words('french')

    # Expressions régulières pour la suppression
    space_pattern = '\s+'  # Espaces supplémentaires
    special_letters = "[^a-zA-Z#]"  # Caractères spéciaux (sauf #)
    punctuation = r'[^\w\s]'  # Ponctuation

    textes_pretraites = []

    for resume in Txt:
        # Nettoyage du texte
        texte = re.sub(space_pattern, ' ', resume)  # Supprime les espaces supplémentaires
        texte = re.sub(special_letters, ' ', texte)  # Supprime les caractères spéciaux
        texte = re.sub(punctuation, '', texte)  # Supprime la ponctuation

        # Séparation des mots et filtrage
        mots = texte.split()  # Divise le texte en mots individuels
        mots = [mot for mot in mots if mot.isalpha()]  # Conserve uniquement les mots alphabétiques
        mots_filtres = [mot for mot in mots if mot not in sw]  # Supprime les stop words
        mots_minuscules = [mot.lower() for mot in mots_filtres]  # Convertit les mots en minuscules

        # Rejoindre les mots nettoyés
        texte_pretraite = " ".join(mots_minuscules)
        textes_pretraites.append(texte_pretraite)

    return textes_pretraites






# Fonction d'extraction d'e-mails
def email_ID(text):
    """
    Extrait les adresses e-mail valides du texte fourni.

    Args:
        text (str): Le texte à partir duquel extraire les adresses e-mail.

    Returns:
        list: Une liste contenant les adresses e-mail trouvées dans le texte, ou une liste vide si aucune adresse e-mail n'est trouvée.
    """

    # Expression régulière pour les adresses e-mail
    email_pattern = r'[A-Za-z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    regex = re.compile(email_pattern)

    # Convertir le texte en chaîne de caractères (au cas où ce ne serait pas déjà le cas)
    text_str = str(text)

    # Recherche des adresses e-mail
    emails = regex.findall(text_str)

    return emails




               
# Fonction pour trouver des numéros de téléphone
def number(text):
    """
    Extrait les numéros de téléphone potentiels du texte fourni.

    Args:
        text (str): Le texte à partir duquel extraire les numéros de téléphone.

    Returns:
        list: Une liste contenant les numéros de téléphone trouvés dans le texte, ou une liste vide si aucun numéro de téléphone n'est trouvé.
    """

    # Expression régulière pour les numéros de téléphone
    pattern = re.compile(r'(\b(?:\+?\d{1,3}[-.\s]?)?\(?(?:\d{2,3})\)?[-.\s/]?\d{2,5}[-.\s/]?\d{2,5}[-.\s/]?\d{2,5}\b)')

    # Recherche des numéros de téléphone
    matches = pattern.findall(text)

    # Nettoyage et filtrage des numéros
    cleaned_numbers = []
    for match in matches:
        # Supprimer les virgules et les points
        match = re.sub(r'[,.]', '', match)

        # Supprimer les caractères non numériques et les espaces excédentaires
        match = re.sub(r'\D$', '', match).strip()

        # Limiter la longueur à 15 chiffres
        if len(re.sub(r'\D', '', match)) <= 15:
            cleaned_numbers.append(match)

    # Vérifier si le numéro est une année
    for number in cleaned_numbers[:]:
        # Diviser le numéro par des tirets
        parts = number.split('-')

        # Si plus de 3 parties, ignorer
        if len(parts) > 3:
            continue

        # Vérifier si les 4 derniers caractères sont des chiffres et représentent une année valide
        try:
            last_four = parts[-1]
            if last_four.isdigit() and int(last_four) in range(1900, 2100):
                cleaned_numbers.remove(number)
        except:
            pass

    # Supprimer les doublons et renvoyer la liste des numéros
    return list(set(cleaned_numbers))





# Fonction pour supprimer les numéros de téléphone pour extraire l'année d'expérience et le nom d'un candidat
def rm_number(text):
    """
    Supprime les numéros de téléphone potentiels du texte fourni.

    Args:
        text (str): Le texte à partir duquel supprimer les numéros de téléphone.

    Returns:
        str: Le texte sans les numéros de téléphone.
    """

    try:
        # Expression régulière pour les numéros de téléphone
        pattern = re.compile(r'([+(]?\d+[)\-]?[ \t\r\f\v]*[(]?\d{2,}[()\
                              ]?[ \t\r\f\v]*\d{2,}[()\-]?[ \t\r\f\v]*\d*[ \t\r\f\v]*\d*[ \t\r\f\v]*)')

        # Recherche des numéros de téléphone
        matches = pattern.findall(text)

        # Nettoyage et filtrage des numéros (similaire à la fonction number)
        cleaned_numbers = []
        for match in matches:
            match = re.sub(r'[,.]', '', match)
            match = re.sub(r'\D$', '', match).strip()
            if len(re.sub(r'\D', '', match)) <= 15:
                cleaned_numbers.append(match)

        # Vérification des années (similaire à la fonction number)
        for number in cleaned_numbers[:]:
            parts = number.split('-')
            if len(parts) > 3:
                continue
            try:
                last_four = parts[-1]
                if last_four.isdigit() and int(last_four) in range(1900, 2100):
                    cleaned_numbers.remove(number)
            except:
                pass

        # Supprimer les numéros du texte
        for number in cleaned_numbers:
            text = text.replace(number, " ")

        return text

    except:
        # Gère les exceptions potentielles (par exemple, expression régulière invalide)
        return text





# Fonction pour supprimer les emails pour extraire l'année d'expérience et le nom d'un candidat
def rm_email(text):
    """
    Supprime les adresses e-mail potentielles du texte fourni.

    Args:
        text (str): Le texte à partir duquel supprimer les adresses e-mail.

    Returns:
        str: Le texte sans les adresses e-mail.
    """

    try:
        # Expression régulière pour les adresses e-mail
        pattern = re.compile('[\w\.-]+@[\w\.-]+')

        # Recherche des adresses e-mail
        matches = pattern.findall(text)

        # Supprimer les adresses e-mail du texte
        for email in matches:
            text = text.replace(email, " ")

        return text

    except:
        # Gère les exceptions potentielles (par exemple, expression régulière invalide)
        return text



# Fonction pour extraire le nom du candidat
def person_name(text):
    """
    Extrait un nom propre d'une chaîne de texte donnée.

    Args:
        text (str): Le texte d'entrée contenant potentiellement des noms propres.

    Returns:
        str: Le premier nom propre identifié dans le texte, ou une chaîne vide si aucun nom n'est trouvé.
    """
    
    # Tokenise le texte entier en phrases
    Sentences = nltk.sent_tokenize(text)
    t = []
    for s in Sentences:
        # Tokenise les phrases en mots
        t.append(nltk.word_tokenize(s))
        
    # Marque chaque mot avec sa partie du discours (POS)
    words = [nltk.pos_tag(token) for token in t]
    n = []
    for x in words:
        for l in x:
            # Vérifie si la balise POS correspond à un nom commun (NN.*)
            if re.match('[NN.*]', l[1]):
                n.append(l[0])
                
    cands = []
    for nouns in n:
        # Si le mot n'est pas trouvé dans le dictionnaire WordNet, on le considère comme un nom propre candidat
        if not wordnet.synsets(nouns):
            cands.append(nouns)
            
    # Retourne le premier nom propre candidat trouvé
    cand = ' '.join(cands[:1])
    return cand





# Fonction pour extraire des années d'expérience
def extract_years_of_experience(resume_text):
    """
    Extrait le total des années d'expérience à partir du texte d'un CV.

    Args:
        resume_text (str): Le texte du CV contenant les dates d'expérience professionnelle.

    Returns:
        int: Le total des années d'expérience calculé à partir des informations fournies.
    """
    
    # Définissez un modèle d'expression régulière pour extraire les plages de dates (par exemple, "(2010 - 2015)" ou "(2010 - présent)")
    pattern = re.compile(r'\((\d{4})\s*[-–]\s*(\d{4}|présent|present|actuel)\)', re.IGNORECASE)
    
    # Trouvez toutes les correspondances dans le texte du CV
    matches = pattern.findall(resume_text)
    
    total_experience = 0
    current_year = datetime.utcnow().year

    for start, end in matches:
        # Gérer différentes représentations du "présent"
        if end.lower() in ['présent', 'present', 'actuel']:
            end = str(current_year)
        
        # Convertir les années de début et de fin en entiers
        start_year = int(start)
        end_year = int(end)

        # Assurez-vous que l'année de début est inférieure ou égale à l'année de fin
        if start_year <= end_year:
            total_experience += end_year - start_year

    return total_experience

        


# Fonction pour extraire l'emplacement
def location(text):
    """
    Extrait les noms de villes à partir d'un texte donné.

    Args:
        text (str): Le texte d'entrée contenant potentiellement des noms de lieux.

    Returns:
        list: Une liste contenant les noms des villes extraites du texte.
    """
    
    # Utilise locationtagger pour trouver les entités de localisation dans le texte
    place_entity = locationtagger.find_locations(text=text)
    
    # Retourne la liste des villes identifiées
    return place_entity.cities






# Fonction pour Extrair les noms d'entreprises 
def CompanyName(text):
    """
    Extrait les noms d'entreprises à partir d'un texte donné.

    Args:
        text (str): Le texte d'entrée contenant potentiellement des noms d'entreprises.

    Returns:
        list: Une liste contenant les noms des entreprises extraites du texte (les deux dernières entreprises si disponibles), ou None si aucune entreprise n'est trouvée.
    """
    
    # Tokeniser le texte en phrases
    sentences = sent_tokenize(text)

    # Initialiser une liste vide pour stocker les noms d'entreprises
    company_names = []

    # Itérer sur chaque phrase
    for sentence in sentences:
        # Analyser la phrase avec spaCy
        doc = nlp(sentence)

        # Identifier les entités nommées de type "ORG" (organisation)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                # Extraire le nom de l'entreprise et l'ajouter à la liste
                company_names.append(ent.text)

    # Nettoyer les noms d'entreprises (supprimer les espaces inutiles, etc.)
    cleaned_company_names = []
    for name in company_names:
        cleaned_name = name.strip().lower()
        if cleaned_name not in cleaned_company_names:
            cleaned_company_names.append(cleaned_name)

    # Vérifiez de nouveau chaque nom d'entreprise nettoyé
    cleaned_company_names2 = []
    for name in cleaned_company_names:
        doc = nlp(name)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                cleaned_company_names2.append(ent.text)

    # Retourner les deux derniers noms d'entreprises nettoyés si disponibles
    if len(cleaned_company_names2) >= 2:
        return cleaned_company_names2[-2:]  
    elif len(cleaned_company_names2) == 1:
        return cleaned_company_names2[-1:]  
    else:
        return None

  





# Fonction d'extraction de compétences
def skills(text):
    """
    Extrait les compétences à partir d'un texte donné en utilisant une liste de compétences prédéfinie.

    Args:
        text (str): Le texte d'entrée contenant potentiellement des mentions de compétences.

    Returns:
        set: Un ensemble contenant les compétences extraites du texte.
    """
    
    # Charger les stopwords français de NLTK
    sw = set(nltk.corpus.stopwords.words('french'))

    # Tokeniser le texte en mots
    tokens = nltk.tokenize.word_tokenize(text)

    # Filtrer les tokens pour ne garder que les mots alphabétiques
    ft = [w for w in tokens if w.isalpha()]

    # Filtrer les tokens pour enlever les stopwords
    ft = [w for w in ft if w not in sw]

    # Générer des bigrammes et trigrammes à partir des tokens filtrés
    n_grams = list(map(' '.join, nltk.everygrams(ft, 2, 3)))

    # Initialiser un ensemble pour stocker les compétences trouvées
    fs = set()

    # Vérifier chaque token contre la liste de compétences et ajouter s'il correspond
    for token in ft:
        if token.lower() in skill:
            fs.add(token)
    
    # Vérifier chaque n-gramme contre la liste de compétences et ajouter s'il correspond
    for ngram in n_grams:
        if ngram.lower() in skill:
            fs.add(ngram)
    
    # Retourner l'ensemble des compétences trouvées
    return fs





# Chargement de compétences prédéfinies
# Charger les compétences à partir d'un fichier Excel
Skills = pd.read_excel('C:/Users/naji/Desktop/3D smart factory/project/data/Skills.xlsx')

# Convertir le DataFrame en une liste de valeurs
Skills = Skills.values.flatten().tolist()

# Convertir tous les éléments de la liste en minuscules
skill = [z.lower() for z in Skills]





# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




# Routes de l'application Flask



# Route principale 
@app.route('/')
def index():
    return render_template('index.html')







# Route pour l'inscription
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """
    Gère la route d'inscription des utilisateurs.

    Si la méthode de requête est POST (formulaire d'inscription soumis), cette fonction :
    - Récupère le nom d'utilisateur et le mot de passe saisis par l'utilisateur
    - Vérifie si le nom d'utilisateur existe déjà dans la base de données
    - Si le nom d'utilisateur n'existe pas, elle hash le mot de passe et crée un nouvel utilisateur dans la base de données
    - Redirige l'utilisateur vers la page de connexion

    Si la méthode de requête est GET (affichage du formulaire d'inscription), cette fonction
    renvoie le template 'signup.html' pour afficher le formulaire d'inscription.

    Returns:
        La réponse HTTP (soit le template 'signup.html', soit une redirection).
    """
    if request.method == 'POST':
        new_user = request.form['username']
        new_password = request.form['password']
        if new_user and new_password:
            # Vérifiez si le nom d'utilisateur existe déjà dans la base de données
            existing_user = users_collection.find_one({"username": new_user})
            if existing_user:
                flash("Username already exists. Please choose a different username.")
                return redirect(url_for('signup'))
            else:
                hashed_password = hash_password(new_password)
                try:
                    users_collection.insert_one({"username": new_user, "password": hashed_password})
                    flash("Account created successfully! Please login.")
                    return redirect(url_for('login'))
                except errors.ServerSelectionTimeoutError as err:
                    flash(f"Failed to connect to MongoDB: {err}")
        else:
            flash("Please fill out both fields.")
    return render_template('signup.html')








# Route pour la connexion
@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Gère la route de connexion des utilisateurs.

    Si la méthode de requête est POST (formulaire de connexion soumis), cette fonction :
    - Récupère le nom d'utilisateur et le mot de passe saisis par l'utilisateur
    - Vérifie si l'utilisateur existe dans la base de données et si le mot de passe est correct
    - Si les informations de connexion sont valides, elle stocke les informations de l'utilisateur dans la session
    - Redirige l'utilisateur vers la page d'upload

    Si la méthode de requête est GET (affichage du formulaire de connexion), cette fonction
    renvoie le template 'login.html' pour afficher le formulaire de connexion.

    Returns:
        La réponse HTTP (soit le template 'login.html', soit une redirection).
    """
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            user = users_collection.find_one({"username": username})
            if user and verify_password(user["password"], password):
                session['logged_in'] = True
                session['username'] = username
                return redirect(url_for('upload'))
            else:
                flash("Incorrect Username/Password")
        except errors.ServerSelectionTimeoutError as err:
            flash(f"Failed to connect to MongoDB: {err}")
    return render_template('login.html')








# Route pour télécharger et de traiter des fichiers en même temps
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    Gère la route d'upload et de traitement des fichiers (CV et descriptions de poste).

    Si l'utilisateur n'est pas connecté, il est redirigé vers la page de connexion.

    Si la méthode de requête est POST (fichiers uploadés), cette fonction :
    - Récupère les fichiers de CV et de descriptions de poste uploadés
    - Lit et prétraite le contenu des fichiers
    - Calcule la similarité cosinus entre les CV et les descriptions de poste
    - Extrait des informations supplémentaires des CV (numéro de téléphone, email, nom, expérience, compétences, localisation, entreprises)
    - Stocke les 3 meilleurs CV correspondant à chaque description de poste dans la base de données
    - Redirige l'utilisateur vers la page des résultats

    Si la méthode de requête est GET (affichage du formulaire d'upload), cette fonction
    renvoie le template 'upload.html' pour afficher le formulaire d'upload.

    Returns:
        La réponse HTTP (soit le template 'upload.html', soit une redirection).
    """

    if 'logged_in' not in session:
        return redirect(url_for('login'))
    

    if request.method == 'POST':
        
        resumes = request.files.getlist('resumes')
        job_descriptions = request.files.getlist('job_descriptions')



        if resumes and job_descriptions:

            flash(f"{len(resumes)} resumes and {len(job_descriptions)} job descriptions uploaded successfully.")
            
            # Lire les CVs
            resumeTxt = read_files(resumes)
            # Lire les descriptions de poste
            jdTxt = read_files(job_descriptions)
            
            # Prétraitement des CVs
            p_resumeTxt = preprocessing(resumeTxt)
            # Prétraitement des descriptions de poste
            jds = preprocessing(jdTxt)
            
            # Combinaison de CV et de description de poste pour le calcul du TF-IDF et de la similarité cosinus
            TXT = p_resumeTxt + jds
            
            # Trouver le score TF-IDF de tous les CV et descriptions de poste
            tv = TfidfVectorizer(max_df=0.8, min_df=1, ngram_range=(1, 3))
            tfidf_wm = tv.fit_transform(TXT)
            tfidf_tokens = tv.get_feature_names_out()
            
            # Conversion de la matrice TF-IDF en DataFrame
            df_tfidfvect1 = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)


            # Calcul de la similarité cosinus entre les descriptions de poste et les CV pour savoir quel CV convient le mieux à une description de poste
            # similarity = cosine_similarity(df_tfidfvect1[0:len(resumeTxt)], df_tfidfvect1[len(resumeTxt):])
            # similarity = (similarity * 100).round(2)
            # jd_names = [os.path.splitext(uploaded_file.filename)[0] for uploaded_file in job_descriptions]
            # Data = pd.DataFrame(similarity, columns=jd_names)
            # t = pd.DataFrame({'Original Resume': resumeTxt})
            # dt = pd.concat([Data, t], axis=1)

            

                        # **Ajout de la réduction de dimensionnalité**
            dimrec = TruncatedSVD(n_components=30, n_iter=7, random_state=42)
            transformed = dimrec.fit_transform(df_tfidfvect1)
            
            # Conversion du vecteur transformé en liste
            vl = transformed.tolist()
            
            # Conversion de la liste en DataFrame
            fr = pd.DataFrame(vl)
            
            # Calcul de la similarité cosinus entre les descriptions de poste et les CV pour savoir quel CV convient le mieux à une description de poste
            similarity = cosine_similarity(fr[0:len(resumeTxt)], fr[len(resumeTxt):])
            similarity = (similarity * 100).round(2)
            jd_names = [os.path.splitext(uploaded_file.filename)[0] for uploaded_file in job_descriptions]
            Data = pd.DataFrame(similarity, columns=jd_names)
            t = pd.DataFrame({'Original Resume': resumeTxt})
            dt = pd.concat([Data, t], axis=1)



            # Appel de la fonction number pour obtenir la liste des numéros des candidats
            dt['Phone No.'] = dt['Original Resume'].apply(lambda x: number(x))

            # Appel de la fonction email_ID pour obtenir la liste des emails des candidats
            dt['E-Mail ID'] = dt['Original Resume'].apply(lambda x: email_ID(x))

            # Appel de la fonction rm_number pour supprimer le numéro de téléphone
            dt['Original'] = dt['Original Resume'].apply(lambda x: rm_number(x))

            # CV avec e-mails et numéros supprimés 
            dt['Original'] = dt['Original'].apply(lambda x: rm_email(x))

            # Appel de la fonction person_name pour extraire le nom d'un candidat
            dt['Candidate\'s Name'] = dt['Original'].apply(lambda x: person_name(x))

            # Appel de la fonction pour extraire l'année d'expérience du candidat
            dt['Experience'] = dt['Original'].apply(lambda x: extract_years_of_experience(x))

            # Appeler la fonction skills pour extraire les compétences d'un candidat
            dt['Skills'] = dt['Original'].apply(lambda x: skills(x))

            # Appel de la fonction location pour extraire la localisation d'un candidat
            dt['Location'] = dt['Original'].apply(lambda x: location(x))

            # Appel de la fonction CompanyName pour extraire les anciennes sociétés d'un candidat
            dt['Company Name'] = dt['Original Resume'].apply(lambda x: CompanyName(x))

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(dt[['Candidate\'s Name', 'Phone No.', 'E-Mail ID', 'Skills', 'Experience', 'Location', 'Company Name']])  
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")



            for jd in jd_names:
                collection = db[jd]
                # Effacer la collection existante avant d'insérer de nouvelles données
                collection.delete_many({})

                top_resumes = dt[[jd, 'Candidate\'s Name', 'Phone No.', 'E-Mail ID', 'Skills', 'Experience', 'Location', 'Company Name', 'Original Resume']].sort_values(by=jd, ascending=False).head(3)
                top_resumes['Skills'] = top_resumes['Skills'].apply(lambda x: list(x) if isinstance(x, set) else x)
                collection.insert_many(top_resumes.to_dict('records'))

            session['jd_names'] = jd_names


            return redirect(url_for('results'))


    return render_template('upload.html')







# Route pour afficher les résultats      
@app.route('/results', methods=['GET', 'POST'])
def results():
    """
    Gère la route d'affichage des résultats de l'appariement entre les CV et les descriptions de poste.

    Si l'utilisateur n'est pas connecté, il est redirigé vers la page de connexion.

    Cette fonction :
    - Récupère la liste des descriptions de poste stockées dans la session
    - Récupère la description de poste sélectionnée par l'utilisateur dans le formulaire
    - Récupère les 3 meilleurs CV correspondant à la description de poste sélectionnée depuis la base de données
    - Génère une image de nuage de mots pour le premier CV
    - Renvoie le template 'results.html' avec les données nécessaires pour l'affichage des résultats

    Returns:
        La réponse HTTP (le template 'results.html').
    """
    # Vérifie si l'utilisateur est connecté, sinon le redirige vers la page de connexion
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    

    # Récupère la liste des descriptions de poste stockées dans la session
    jd_names = session.get('jd_names', [])

    # Récupère la description de poste sélectionnée par l'utilisateur dans le formulaire
    selected_jd = request.form.get('jd_select')

    # Initialise les variables pour stocker les résultats et le chemin de l'image du nuage de mots
    top_resumes = None
    wordcloud_image_path = None

    # Si une description de poste est sélectionnée
    if selected_jd:
        # Récupère les 3 meilleurs CV correspondant à la description de poste sélectionnée depuis la base de données
        collection = db[selected_jd]
        top_resumes = list(collection.find())

        # Génère et enregistre l'image du nuage de mots pour le premier CV
        if top_resumes and 'Original Resume' in top_resumes[0]:
            resume_text = top_resumes[0]['Original Resume']
            wordcloud = WordCloud(width=800, height=500, background_color='white', min_font_size=10).generate(resume_text)
            wordcloud_image_path = os.path.join('static', 'wordcloud.png')
            wordcloud.to_file(wordcloud_image_path)

    # Renvoie le template 'results.html' avec les données nécessaires pour l'affichage des résultats
    return render_template('results.html', jd_names=jd_names, top_resumes=top_resumes, selected_jd=selected_jd, wordcloud_image_path=wordcloud_image_path)






# Route pour la déconnexion
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))





if __name__ == '__main__':
    app.run(debug=True)
