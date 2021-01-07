from flask import Flask,render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('static\LinearRegressionModel.pkl','rb'))
df = pd.read_csv('static\Clean_data.csv')
df = df.drop('Unnamed: 0',axis=1)


@app.route('/',methods=['POST','GET'])
def home():
    area_type = sorted(df['area_type'].unique())
    location = sorted(df['location'].unique())
    bath = sorted(df['bath'].unique())
    balcony = sorted(df['balcony'].unique())
    BHK = sorted(df['BHK'].unique())

    return render_template('index.html',area_types=area_type, locations=location, baths=bath, balconies=balcony, BHK=BHK)


@app.route('/predict',methods=['POST','GET'])
def pred():
    if request.method == 'POST':

        area_type = request.form.get('a_type')
        location = request.form.get('loc')
        total_sqft = float(request.form.get('total_sqft'))
        bath = float(request.form.get('bath_room'))
        balcony = float(request.form.get('my_balcony'))
        bhk = int(request.form.get('bhk'))

        values = model.predict(pd.DataFrame([[area_type,location,total_sqft,bath,balcony,bhk]],columns=['area_type', 'location', 'total_sqft', 'bath', 'balcony', 'BHK']))
        values = np.round(values,2)[0]
        return render_template('predictions.html',values=values)



if __name__ == "__main__":
    app.run(debug=True)