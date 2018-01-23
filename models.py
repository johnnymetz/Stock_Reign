# MODELS FOR SQL DATABASE
from app import db
from datetime import datetime


# Stock History
class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), unique=True, nullable=False)
    price_history = db.Column(db.PickleType, nullable=False)
    date_stored = db.Column(db.DateTime)
    # created_on = db.Column(db.DateTime, default=datetime.utcnow)
    # last_modified = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return '<{}>'.format(self.ticker)
