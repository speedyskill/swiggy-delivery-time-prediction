FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

COPY requirements-dockers.txt ./

RUN pip install -r requirements-dockers.txt

COPY Swiggy-logo.png ./Swiggy-logo.png
COPY app.py ./
COPY frontend.py ./

COPY ./models/preprocessor.joblib ./models/preprocessor.joblib
COPY ./scripts/data_clean_utils.py ./scripts/data_clean_utils.py
COPY ./run_information.json ./

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & \
                  streamlit run frontend.py --server.port=8501 --server.address=0.0.0.0"]



