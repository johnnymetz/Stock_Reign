from app import db
from models import Stock
import pandas_datareader as pdr
import datetime


existing_tickers = [obj.ticker for obj in Stock.query.all()]

for ticker in ['box', 'fb', 'panw', 'amzn']:
    ticker = ticker.upper()
    try:
        price_history = pdr.get_data_yahoo(ticker)
    except pdr._utils.RemoteDataError:
        print('--> Unable to fetch data for {}'.format(ticker))
        continue
    now = datetime.datetime.now()
    # Update record if already exists
    if ticker in existing_tickers:
        stock = Stock.query.filter_by(ticker=ticker).first()
        stock.price_history = price_history
    # Create record if new
    else:
        stock = Stock(
            ticker=ticker,
            price_history=price_history
        )
    db.session.add(stock)
    print('{} successfully added.'.format(ticker))

db.session.commit()
