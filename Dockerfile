# Utiliser une image Python légère
FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers du projet dans le conteneur
COPY requirements.txt /app/
COPY app /app/
COPY data /data/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Définir le point d’entrée par défaut
CMD ["bash"]
