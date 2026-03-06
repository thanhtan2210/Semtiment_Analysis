PYTHON := python

.PHONY: help train serve compose-up

help:
	@echo "Available targets:"
	@echo "  make train       - Train model locally"
	@echo "  make serve       - Run API locally (uvicorn)"
	@echo "  make compose-up  - Start MLflow + API via docker-compose"

train:
	$(PYTHON) train.py --input sentimentdataset.csv --normalize-labels

serve:
	uvicorn serve:app --reload --port 8000

compose-up:
	docker-compose up --build
