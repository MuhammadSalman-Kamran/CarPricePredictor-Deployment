import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, Prediction

df = pd.read_csv('artifacts/sample.csv')
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        unique_model = sorted(df['name'].unique())
        company = sorted(df['company'].unique())
        year = sorted(df['year'].unique(), reverse= True)
        type = df['fuel_type'].unique()
        company.insert(0,'Select Company')
        return render_template('index.html', companies = company, models = unique_model, years = year, fuel_types = type)
    
    else:
        custom_data_obj = CustomData(
            company = request.form.get('company'),
            selected_model = request.form.get('model'),
            year = int(request.form.get('year')),
            type = request.form.get('fuel_type'),
            kilo = int(request.form.get('kilo_driven')))
        input_df = custom_data_obj.data_as_df()

        prediction_obj = Prediction()
        prediction_value = prediction_obj.prediction(input_df)
        
        return render_template('index.html', prediction = np.round(prediction_value[0], 0))


if __name__ == '__main__':
    app.run(debug=True)