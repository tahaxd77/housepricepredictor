from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        property_type = request.form['property_type']
        print(property_type)
        area_size = int(request.form['area_size'])
        area_type = request.form['area_type']
        rooms = int(request.form['rooms'])
        purpose = request.form['purpose']
        location = request.form['location']
        city = request.form['city']
    
        price = calculate_prices(property_type, area_size, area_type, rooms, purpose, location, city)
        price = price.round(2)
        return render_template('result.html', price=price)
    return render_template('index.html')

def calculate_prices(property_type, area_size, area_type, rooms, purpose, location, city):
    # Load the model
    model = pickle.load(open('model.pkl', 'rb'))
    column_transformer = pickle.load(open('preprocessor.pkl', 'rb'))

    # Create a dataframe
    data = {'property_type': [property_type], 'Area Size': [area_size], 'Area Type': [area_type], 'Rooms': [rooms], 'purpose': [purpose], 'location': [location], 'city': [city]}
    print(data)
    df = pd.DataFrame(data)

    # Transform the data
    df_transformed = column_transformer.transform(df)

    # Predict the price
    price = model.predict(df_transformed)
    return price

if __name__ == '__main__':
    app.run(debug=True)