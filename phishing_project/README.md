Phishing Attack Protection System (Random Forest + Flask Web App)
=================================================================
Project skeleton for Final Year Project (FYP)

Structure:
- data/: put your labelled dataset CSV files here (e.g. phishing_dataset.csv)
- models/: trained model will be saved here (phishing_rf.pkl)
- src/: source code (feature extractor, training, evaluation, Flask app)
- templates/: Flask HTML templates
- static/: static assets (CSS)
- requirements.txt: python dependencies

Quick start:
1. Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt

2. Prepare dataset:
   Place a CSV file at data/phishing_dataset.csv with columns: url,label
   label: 1 = phishing, 0 = legitimate

3. Train (this will extract features and save the model):
   python src/train.py

4. Run the Flask demo:
   python src/app.py
   Open http://127.0.0.1:5000 in your browser.

Notes:
- Feature extraction performs WHOIS, SSL checks and HTML fetches; it's network-dependent and may be slow.
- For initial testing, use a small sample of the dataset (200-1000 rows).
- See src/feature_extractor.py for details of features and handling of missing values.
