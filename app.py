# Paquetes de Flask
from flask import Flask,render_template,request,url_for
from flask_bootstrap import Bootstrap 
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from flask_sqlalchemy import SQLAlchemy 

import os
import datetime
import time


# Paquetes de Analisis de Datos
import pandas as pd 
import numpy as np 

# Paquetes de ML
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Paquetes de ML para vectorización de texto y extracción de caracteristicas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer




app = Flask(__name__)
Bootstrap(app)

# Configuración para los archivos subidos
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
configure_uploads(app,files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////data/filestorage.db'
db = SQLAlchemy(app)


# Guardando los datos en las Base de Datos 
class FileContents(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	name = db.Column(db.String(300))
	modeldata = db.Column(db.String(300))
	data = db.Column(db.LargeBinary)


@app.route('/')
def index():
	return render_template('index.html')

# Ruta para nuestro procesamiento y pagina de detalle
@app.route('/dataupload',methods=['GET','POST'])
def dataupload():
	if request.method == 'POST' and 'csv_data' in request.files:
		file = request.files['csv_data']
		filename = secure_filename(file.filename)
		#os.path.join es usado para que trabaje en cualquier sistema operativo
        #file.save(os.path.join("wherever","you","want",filename))
		file.save(os.path.join('data/',filename))
		fullfile = os.path.join('data/',filename)

		# Para el tiempo
		date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

		# Funcion para el analisis exploratorio de datos
		df = pd.read_csv(os.path.join('data/',filename))
		df_size = df.size
		df_shape = df.shape
		df_columns = list(df.columns)
		df_targetname = df[df.columns[-1]].name
		df_featurenames = df_columns[0:-1] # select all columns till last column
		df_Xfeatures = df.iloc[:,0:-1] 
		df_Ylabels = df[df.columns[-1]] # Select the last column as target
		# same as above df_Ylabels = df.iloc[:,-1]
		

		# Construcción del modelo
		X = df_Xfeatures
		Y = df_Ylabels
		seed = 7
		# prepare models
		models = []
		models.append(('LR', LogisticRegression()))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('KNN', KNeighborsClassifier()))
		models.append(('CART', DecisionTreeClassifier()))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC()))
		# Evaluando cada modelo por turno
		

		results = []
		names = []
		allmodels = []
		scoring = 'accuracy'
		for name, model in models:
			kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
			cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
			results.append(cv_results)
			names.append(name)
			msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
			allmodels.append(msg)
			model_results = results
			model_names = names 
			
		# Guardando Resultados de los archivos subidos en Sqlite DB
		newfile = FileContents(name=file.filename,data=file.read(),modeldata=msg)
		db.session.add(newfile)
		db.session.commit()		
		
	return render_template('details.html',filename=filename,date=date,
		df_size=df_size,
		df_shape=df_shape,
		df_columns =df_columns,
		df_targetname =df_targetname,
		model_results = allmodels,
		model_names = names,
		fullfile = fullfile,
		dfplot = df
		)




if __name__ == '__main__':
	app.run(debug=True)





# Jesus Saves @ JCharisTech