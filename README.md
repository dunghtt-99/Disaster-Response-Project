# Disaster Response Pipeline Project
This project is about to analyze messages in disasters and divide them into categories that can be viewes in a web app. This will be a helpful application for community to have a wide vision about disaster situations. It will also help people or organizations have better preparations for disasters in the future.

## Development Language and Libraries
Python, Scikit Learn, Pandas, NumPy, SQLalchemy, NLTK, Pickle

## Folder Structure
- app
	- template
		- master.html
		- go.html
	- run.py
- data
	- disaster_categories.csv
	- disaster_messages.csv
	- process_data.py
	- DisasterResponse.db
- models
	- train_classifier.py
- README.md

## Repository
https://github.com/dunghtt-99/Disaster-Response-Project/

## Instruction
- Step 1: Clone the repo
- Step 2: Run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv`
- Step 3: Run ML pipeline that trains classifier and saves
        `python models/train_classifier.py models/classifier.pkl`
- Step 4: Run web app
	`python app/run.py`
- Step 5: Access web app on localhost:3000

## Acknowledgement
https://www.udacity.com/
