#app/models.py

from app import db

class Customer(db.Model):
    __tablename__ = 'customers'
    customer_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))

class Product(db.Model):
    __tablename__ = 'products'
    product_id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(100))
    name = db.Column(db.String(100))

class Order(db.Model):
    __tablename__ = 'orders'
    order_id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.customer_id'))
    product_id = db.Column(db.Integer, db.ForeignKey('products.product_id'))
    purchase_date = db.Column(db.DateTime)

class Recommendation(db.Model):
    __tablename__ = 'recommendations'
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.customer_id'))
    algorithm = db.Column(db.String(50))
    recommendations = db.Column(db.String(500))
    probability = db.Column(db.Float)

class OptimalInterval(db.Model):
    __tablename__ = 'optimal_intervals'
    customer_id = db.Column(db.Integer, primary_key=True)
    optimal_days_between_purchases = db.Column(db.Float)
