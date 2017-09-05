from flask import Flask, request, session, render_template, redirect, url_for, flash
import pandas_datareader as pdr
import pandas as pd
import datetime


app = Flask(__name__)
app.config['SECRET_KEY'] = 'ReignSupreme'


@app.route('/', methods=['GET', 'POST'])
def index():

    # Build DataFrame
    if 'ticker' not in session:
        session['all_tickers'] = ['box', 'panw', 'v', 'amzn', 'fb', 'wix']
        session['ticker'] = 'fb'
        session['range'] = {'start': datetime.datetime(2017, 3, 1), 'end': datetime.datetime.today()}
    df = pd.read_pickle('data/pickles/{}.pickle'.format(session['ticker']))
    start = session['range']['start']
    end = session['range']['end']
    df = df.loc[start:end][['Open', 'Close', 'Volume']]

    def calculate_pct_change(row):
        pct_change = (row['Close'] - row['Open']) / row['Open']
        return pct_change

    def get_pct_change_category(row):
        if row['pct_change'] > 0:
            return 'positive'
        elif row['pct_change'] < 0:
            return 'negative'
        else:
            return 'default'

    df['pct_change'] = df.apply(func=calculate_pct_change, axis=1)
    df['pct_change_category'] = df.apply(func=get_pct_change_category, axis=1)

    # Extract data
    stats = df.describe().to_dict()
    stats['min_pct_change_category'] = get_pct_change_category(row={'pct_change': df['pct_change'].min()})
    stats['max_pct_change_category'] = get_pct_change_category(row={'pct_change': df['pct_change'].max()})
    most_recent_date = df.index[-1]
    df_pct_change = (df.loc[most_recent_date, 'Close'] - df.loc[start, 'Open']) / df.loc[
        start, 'Open']
    df_total_days = (most_recent_date - start).days
    stats['total_pct_change'] = df_pct_change
    stats['total_pct_change_category'] = get_pct_change_category(row={'pct_change': stats['total_pct_change']})
    stats['total_days'] = df_total_days
    stats['avg_pct_change_category'] = get_pct_change_category(row={'pct_change': df['pct_change'].mean()})
    raw_data = df.reset_index().to_dict(orient='records')

    # Chart data
    dates = [i.strftime('%Y-%m-%d') for i in df.index.tolist()]
    prices = df.Close.tolist()
    dates.insert(0, 'x')
    prices.insert(0, session['ticker'].upper())
    chart_data = {
        'dates': dates,
        'prices': prices
    }

    return render_template('stock_index.html', raw_data=raw_data, stats=stats, chart_data=chart_data)


@app.route('/edit_dates', methods=['POST'])
def edit_dates():
    start_date = datetime.datetime.strptime(request.form['start_date'], '%m/%d/%Y')
    end_date = datetime.datetime.strptime(request.form['end_date'], '%m/%d/%Y')
    df = pd.read_pickle('data/pickles/{}.pickle'.format(session['ticker']))
    if start_date not in df.index.tolist() or end_date not in df.index.tolist():
        flash('Data not found for the selected dates. Try again with at least one different date.', category='danger')
        return redirect(url_for('index'))
    session['range']['start'] = start_date
    session['range']['end'] = end_date
    flash('Dates successfully updated.', category='success')
    return redirect(url_for('index'))


@app.route('/change_ticker', methods=['POST'])
def change_ticker():
    session['ticker'] = request.form['ticker']
    flash('Ticker successfully changed.', category='success')
    return redirect(url_for('index'))


@app.route('/add_ticker', methods=['POST'])
def add_ticker():
    ticker = request.form['ticker'].lower()
    session['ticker'] = ticker
    session['all_tickers'].append(ticker)
    stock_data = pdr.get_data_yahoo(ticker.upper())
    stock_data.to_pickle('pickles/stocks/{}.pickle'.format(ticker))
    return redirect(url_for('index'))


@app.route('/update_tickers', methods=['GET', 'POST'])
def update_tickers():
    for ticker in session['all_tickers']:
        stock_data = pdr.get_data_yahoo(ticker.upper())
        stock_data.to_pickle('data/pickles/{}.pickle'.format(ticker))
    flash('Ticker data successfully updated.', category='success')
    return redirect(url_for('index'))


@app.route('/clear')
def clear():
    session.clear()
    flash('Data successfully reset.', category='success')
    return redirect(url_for('index'))





if __name__ == '__main__':
    app.run(debug=True)
