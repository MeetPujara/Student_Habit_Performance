#base image
FROM python:3.12.6

#workdir
WORKDIR /app

#copy
COPY . /app

#run 
RUN pip install -r requirements.txt

#port
EXPOSE 8501

#command
CMD ["streamlit", "run", "Frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]