# mental-health-counselors

## To run the app
 go to dev branch
 
1. pip3 install -r requirements.txt
2. add the openAI key
3. uvicorn app.main:app --reload --port 8000

## To ingest data:
 1. In postman in http://localhost:8000/ingest-csv
 2. headers: Content-Type - application/json
 3. body: form-data
 4. value: train.csv
 5. <img width="661" height="346" alt="Captura de pantalla 2025-09-03 a la(s) 4 53 51 p m" src="https://github.com/user-attachments/assets/5c33d8bc-40a3-47d8-b63f-9253b9d42581" />

 ## to ask:
 http://localhost:8000/ask
 
<img width="890" height="246" alt="Captura de pantalla 2025-09-03 a la(s) 4 57 38 p m" src="https://github.com/user-attachments/assets/14330d4e-f4a3-404b-842b-0b946885bbeb" />

should get:
 id: 
score: 
question: 
answer: 
source: 
tags: 
