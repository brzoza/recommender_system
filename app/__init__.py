from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import logging
import urllib

app = Flask(__name__)

# Ustawienia połączenia do bazy danych Azure SQL
params = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};"
                                 "SERVER=<your_server>.database.windows.net;"
                                 "DATABASE=<your_database>;"
                                 "UID=<your_username>;"
                                 "PWD=<your_password>")
app.config['SQLALCHEMY_DATABASE_URI'] = f"mssql+pyodbc:///?odbc_connect={params}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Konfiguracja loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('app')
file_handler = logging.FileHandler('logs/app.log')
error_handler = logging.FileHandler('logs/errors.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(error_handler)

from app.routes import main
app.register_blueprint(main)
