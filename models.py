# MODELS FOR SQL DATABASE
from app import db


# Stock History
class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), unique=True, nullable=False)
    price_history = db.Column(db.PickleType, nullable=False)
    date_stored = db.Column(db.DateTime)

    def __repr__(self):
        return '<{}>'.format(self.ticker)
