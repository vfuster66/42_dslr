# Nom de l'image Docker
IMAGE_NAME = sorting-hat
WORKDIR = /app

# Fichiers et scripts
TRAIN_SCRIPT = app/logreg_train.py
PREDICT_SCRIPT = app/logreg_predict.py
DESCRIBE_SCRIPT = app/describe.py
HISTOGRAM_SCRIPT = app/histogram.py
SCATTER_SCRIPT = app/scatter_plot.py
PAIRPLOT_SCRIPT = app/pair_plot.py
EVALUATE_SCRIPT = app/logreg_evaluate.py

# Dossiers et fichiers CSV
DATA_DIR = /data
TRAIN_DATA = $(DATA_DIR)/dataset_train.csv
TEST_DATA = $(DATA_DIR)/dataset_test.csv
PREDICTION_OUTPUT = $(DATA_DIR)/houses.csv

# ---------------------------------------------
# CONSTRUCTION & SHELL
# ---------------------------------------------

# Construire l'image Docker
build:
	@echo "🛠️  Build de l'image Docker $(IMAGE_NAME)..."
	@docker build -t $(IMAGE_NAME) .

# Lancer un conteneur interactif
shell:
	@echo "🐚 Lancement du shell interactif dans Docker..."
	@docker run -it --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) bash

# ---------------------------------------------
# EXÉCUTION DES SCRIPTS
# ---------------------------------------------

# Description des données
describe:
	@echo "📊 Exécution de describe.py..."
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 $(DESCRIBE_SCRIPT) $(TRAIN_DATA)

histogram:
	@echo "📊 Génération des histogrammes..."
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 $(HISTOGRAM_SCRIPT) $(TRAIN_DATA)

scatter:
	@echo "📈 Génération des scatter plots..."
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 $(SCATTER_SCRIPT) $(TRAIN_DATA)

pairplot:
	@echo "🔗 Génération du pair plot..."
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 $(PAIRPLOT_SCRIPT) $(TRAIN_DATA)

train:
	@echo "🏋️  Lancement de l'entraînement interactif\n"
	@docker run --rm -it -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) \
	python3 $(TRAIN_SCRIPT)

predict:
	@echo "🔮 Prédictions sur le dataset de test..."
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 $(PREDICT_SCRIPT) $(TEST_DATA)

evaluate:
	@echo "📏 Évaluation sur le dataset d'entraînement..."
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 $(EVALUATE_SCRIPT) $(TRAIN_DATA)

roc:
	@echo "📈 Génération des courbes ROC pour le dataset d'entraînement..."
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 app/logreg_roc.py $(TRAIN_DATA)

roc-all:
	@echo "📈 Génération des courbes ROC pour train et test..."
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 app/logreg_roc.py $(TRAIN_DATA) $(TEST_DATA)

# ---------------------------------------------
# TESTS
# ---------------------------------------------

# Exécuter les tests unitaires
test:
	@echo "🧪 Exécution des tests unitaires..."
	@docker run --rm -it -v $(PWD):/app -w /app $(IMAGE_NAME) python3 -m unittest discover -s tests -p "test_*.py"

# Exécuter un fichier de test spécifique
test-file:
	@echo "🧪 Exécution des tests sur le fichier $(FILE)..."
	@docker run --rm -it -v $(PWD):/app -w /app $(IMAGE_NAME) \
	python3 -m unittest -v $(FILE)


# ---------------------------------------------
# CLEAN
# ---------------------------------------------

# Nettoyer les fichiers générés
clean:
	@echo "🧹 Nettoyage des fichiers générés..."
	rm -f $(PREDICTION_OUTPUT) data/*.png

# Réinitialiser l'environnement sans supprimer les datasets
reset: clean build

# Tout supprimer et tout reconstruire (y compris l'image Docker)
hard-reset:
	@echo "💣 Réinitialisation complète..."
	rm -f data/*.png
	docker rmi -f $(IMAGE_NAME)
	make build
