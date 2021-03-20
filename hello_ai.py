from tensorflow import keras
import numpy as np
import locale

# Build a neural network that predicts the price of a house according to a simple formula.
# So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.
# Create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.
# the house prices are scaled down. So the prediction is in 'hundreds of thousands'


def house_model(x_bed: int) -> float:
    """Predict the house prices

    A neural model that predicts house prices based on already known bedroom - price data set

    :param x_bed: number of bedrooms for which price needs to be predicted
    :return: [description]
    :rtype: [type]
    """
    bed_data = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
    price_data = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(bed_data, price_data, epochs=500)
    return float(model.predict(x_bed))


if __name__=='__main__':
    print('House price prediction starts')
    bed_rooms = 7
    prediction = house_model([bed_rooms])
    
    
    locale.setlocale(locale.LC_ALL, '')
    # round to 7 points since unit is in hundreds of thousands
    predct_usd = locale.currency(round(prediction, 7) * 100000, grouping = True )

    print(f'Predicted house price for {bed_rooms} bed rooms is {predct_usd}')