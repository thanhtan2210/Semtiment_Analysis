PYTHON := python

.PHONY: help train producer processor api dashboard compose-up

help:
	@echo "Available targets:"
	@echo "  make train       - Train model"
	@echo "  make producer    - Run Kafka producer"
	@echo "  make processor   - Run Spark processor"
	@echo "  make api         - Run FastAPI"
	@echo "  make dashboard   - Run Streamlit"
	@echo "  make compose-up  - Start infra via docker-compose"

train:
	$(PYTHON) src/ml/train.py

producer:
	$(PYTHON) src/ingestion/producer.py

processor:
	$(PYTHON) src/processing/processor.py

api:
	uvicorn src.api.main:app --reload --port 8000

dashboard:
	streamlit run src/dashboard/app.py

compose-up:
	docker-compose up --build
