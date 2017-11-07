from app import app
from flask import request, session, render_template, redirect, url_for, flash
from models import Stock
import random
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd


@app.template_filter('get_pct_change_category')
def get_pct_change_category(value):
    if value > 0:
        return 'positive'
    elif value < 0:
        return 'negative'
    else:
        return 'default'


def get_non_weekend(date):
    if date.weekday() == 5:  # Saturday
        return date - datetime.timedelta(days=1)
    elif date.weekday() == 6:  # Sunday
        return date + datetime.timedelta(days=1)
    return date


@app.route('/', methods=['GET', 'POST'])
def index():

    # Get Stocks for DB
    all_tickers = Stock.query.all()
    random_ticker = random.choice(all_tickers).ticker

    # Construct DF for selected stock
    df_all = Stock.query.filter_by(ticker=session.get('current_ticker', random_ticker)).first().price_history
    ipo_date = df_all.index[0]
    latest_date = df_all.index[-1]

    if 'current_ticker' not in session:
        session['current_ticker'] = random_ticker
        session['start_date'] = get_non_weekend(latest_date - relativedelta(months=3))
        session['end_date'] = latest_date

    df = df_all.loc[session['start_date']:session['end_date']][['Adj Close', 'Volume']]
    df['percent_change'] = df['Adj Close'].pct_change(periods=1)  # pct change since close 1 row back

    # Extract Data
    stats = df.describe().to_dict()
    stats['total_days'] = (session['end_date'] - session['start_date']).days
    start_date_price = df.loc[session['start_date'], 'Adj Close']
    end_date_price = df.loc[session['end_date'], 'Adj Close']
    stats['percent_change_range'] = (end_date_price - start_date_price) / start_date_price
    raw_data = df.reset_index().to_dict(orient='records')

    # Chart data
    dates = [i.strftime('%Y-%m-%d') for i in df.index.tolist()]
    prices = df['Adj Close'].tolist()
    dates.insert(0, 'x')
    prices.insert(0, session['current_ticker'])
    chart_data = {
        'dates': dates,
        'prices': prices
    }

    # print(session['start_date'], type(session['start_date']))
    # print(session['end_date'], ipo_date, latest_date)
    # print(stats['total_days'])

    return render_template('stock_index.html', all_tickers=all_tickers, ipo_date=ipo_date, latest_date=latest_date,
                           raw_data=raw_data, stats=stats, chart_data=chart_data)


@app.route('/update_dates', methods=['POST'])
def update_dates():
    dates = [date.strip() for date in request.form['daterange'].split('-')]
    start_date = datetime.datetime.strptime(dates[0], '%m/%d/%Y')
    end_date = datetime.datetime.strptime(dates[1], '%m/%d/%Y')
    df = Stock.query.filter_by(ticker=session['current_ticker']).first().price_history
    if start_date not in df.index.tolist() or end_date not in df.index.tolist():
        flash('Data not found for the selected dates. Please try a different date.', category='danger')
        return redirect(url_for('index'))
    session['start_date'] = start_date
    session['end_date'] = end_date
    flash('Dates successfully updated.', category='success')
    return redirect(url_for('index'))


@app.route('/update_ticker', methods=['POST'])
def update_ticker():
    session['ticker'] = request.form['ticker']
    flash('Ticker successfully changed.', category='success')
    return redirect(url_for('index'))


@app.route('/clear')
def clear():
    session.clear()
    flash('Data successfully reset.', category='success')
    return redirect(url_for('index'))
