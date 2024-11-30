# Doctor56 AI

How to run?


1. Put all your documents under resources/documents (you can include sub-folders)

2. Setup python environment python3.10+
```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3. Process document into computer understable forms
```
python process_documents.py
```
5. Query AI

You need to request an api_key from google's GEMINI AI (be careful of the cost)

https://aistudio.google.com/apikey

change the "query" variable in dctor56_ai.py and run
```
export GEMINI_API_KEY=$MY_KEY
python dctor56_ai.py
```