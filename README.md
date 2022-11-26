# Disaster Response Pipeline Project
This project is about to build a web app for user to view the categories of messages in disasters.

## Development Language and Libraries
Python, Scikit Learn, Pandas, NumPy, SQLalchemy, NLTK, Pickle

## Folder Structure
1. app
1.1. template
1.1.1. master.html
1.1.2. go.html
1.2. run.py
2. data
2.1. disaster_categories.csv
2.2. disaster_messages.csv
2.3. process_data.py
2.4. DisasterResponse.db
3. models
3.1. train_classifier.py
4.README.md

## Repository
https://github.com/dunghtt-99/Disaster-Response-Project/

## Instruction
Step 1: Clone the repo
Step 2: Run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
Step 3: Run ML pipeline that trains classifier and saves
        `python models/train_classifier.py models/classifier.pkl`
Step 4: Run web app
	`python app/trun.py`
Step 5: Access web app on localhost:3000

## Acknowledgement
https://www.udacity.com/
