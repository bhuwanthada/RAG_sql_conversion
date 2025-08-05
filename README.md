## Project Description
This project is for converting user text NLP to sql query. This further hit the tables to process the query and generate records.

## Technical Specification
Python- Fastapi as backend
Streamlit as frontend
sqlite for DB
chromadb for vector store

## Steps to execute
Make sure you have python 3.10 version available on your machine.
Run command to setup virtual env. : ```pyhton3 -m venv <name_of_venv>```
Run the command to install all packages : ```pip install -r req.txt```
Spin up the backend server using command run command: ```uvicorn main:app --reload```
Spin up the frontend streamlit server using command: ```streamlit run st_app.py```
Streamlit open up a page on browser. You can paste your query and system will process it. Post successful completion, system will provide the converted sql query and results generated on top of that.