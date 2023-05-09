FROM python:2.7.18-stretch

WORKDIR /app

COPY ./version_python_2 /app/

RUN apt-get update 
RUN apt-get upgrade -y
RUN apt-get install pandoc texlive-xetex texlive-fonts-recommended -y

RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install notebook

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
