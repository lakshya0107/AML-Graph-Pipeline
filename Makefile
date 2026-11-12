.PHONY: install data train train-dev api dashboard test lint

install:
	pip install -r requirements-dev.txt

data:
	python -m aml_graph.data_pipeline

train: data
	python -m aml_graph.model --backend xgboost

# Development fallback when XGBoost is not installed
train-dev: data
	python -m aml_graph.model --backend sklearn

api:
	uvicorn aml_graph.app:app --reload

dashboard:
	streamlit run dashboard/streamlit_app.py

test:
	pytest

lint:
	ruff check .
