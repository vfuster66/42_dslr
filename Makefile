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

# Dossiers et fichiers CSV
DATA_DIR = /data
TRAIN_DATA = $(DATA_DIR)/dataset_train.csv
TEST_DATA = $(DATA_DIR)/dataset_test.csv
PREDICTION_OUTPUT = $(DATA_DIR)/houses.csv

# Construire l'image Docker
build:
	docker build -t $(IMAGE_NAME) .

# Lancer un conteneur interactif
shell:
	docker run -it --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) bash

# Exécuter les scripts Python
describe:
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 $(DESCRIBE_SCRIPT) $(TRAIN_DATA)

histogram:
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 $(HISTOGRAM_SCRIPT) $(TRAIN_DATA)

scatter:
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 $(SCATTER_SCRIPT) $(TRAIN_DATA)

pairplot:
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 $(PAIRPLOT_SCRIPT) $(TRAIN_DATA)

train:
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 $(TRAIN_SCRIPT) $(TRAIN_DATA)

predict:
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 $(PREDICT_SCRIPT) $(TEST_DATA)

evaluate:
	@docker run --rm -v $(PWD):$(WORKDIR) -v $(PWD)/data:$(DATA_DIR) $(IMAGE_NAME) python3 app/logreg_evaluate.py $(TRAIN_DATA)


# Exécuter les tests unitaires
test:
	@echo "🧪 Exécution des tests..."
	@docker run --rm -it -v $(PWD):/app -w /app $(IMAGE_NAME) python3 -m unittest discover -s tests -p "test_*.py"

# Nettoyer les fichiers générés (NE TOUCHE PAS aux fichiers CSV des datasets !)
clean:
	rm -f $(PREDICTION_OUTPUT) data/*.png

# Réinitialiser l'environnement sans supprimer les datasets
reset: clean build

# Tout supprimer et tout reconstruire (y compris les datasets)
hard-reset:
	rm -f data/*.png
	docker rmi -f $(IMAGE_NAME)
	make build
