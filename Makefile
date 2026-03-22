.PHONY: install train-yolov5 train-yolov8 serve test lint clean

install:
	pip install -r requirements.txt
	pip install -e .

train-yolov5:
	python -m training.train_yolov5 --config config/yolov5_config.yaml

train-yolov8:
	python -m training.train_yolov8 --config config/yolov8_config.yaml

train-emotion:
	python -m training.train_emotion

evaluate:
	python -m training.evaluate

serve:
	python -m api.app

detect:
	python -m inference.webcam --source 0

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	python -m py_compile config/settings.py
	python -m py_compile models/yolov8_detector.py
	python -m py_compile api/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
