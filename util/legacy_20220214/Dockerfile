FROM python:3.10.0

WORKDIR /

RUN pip install pip==21.3.1

#TODO: add all requirements
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "./get_python_details.py" ]

##########
#How to run
#docker build -t test_py_version . --no-cache
#docker run test_py_version
