import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import implicit
from app.models import Order, Product, Customer, Recommendation, OptimalInterval
from app import db, logger
import numpy as np
from scipy.sparse import csr_matrix
import os

def get_data():
    logger.info('Pobieranie danych z bazy danych.')
    orders = db.session.query(Order).all()
    data = [(order.customer_id, order.product_id, order.purchase_date) for order in orders]
    df = pd.DataFrame(data, columns=['customer_id', 'product_id', 'purchase_date'])
    logger.info('Dane pobrane pomyślnie.')
    return df

# Association Rules
def generate_association_rules(df):
    logger.info('Generowanie reguł asocjacyjnych.')
    basket = df.groupby(['customer_id', 'product_id'])['purchase_date'].count().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    logger.info('Reguły asocjacyjne wygenerowane pomyślnie.')
    return rules, basket

def get_association_recommendations(customer_id, rules, basket):
    customer_products = basket.loc[customer_id]
    recommendations = []
    for _, row in rules.iterrows():
        if all(item in customer_products[customer_products > 0].index for item in row['antecedents']):
            recommendations.extend(list(row['consequents']))
    return list(set(recommendations))

# RNN (LSTM)
def prepare_rnn_data(df):
    logger.info('Przygotowywanie danych dla RNN.')
    sequences = df.groupby('customer_id')['product_id'].apply(list).tolist()
    le = LabelEncoder()
    encoded_sequences = [le.fit_transform(seq) for seq in sequences]

    X, y = [], []
    for seq in encoded_sequences:
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])

    X = pad_sequences(X, padding='pre')
    y = np.array(y)
    logger.info('Dane dla RNN przygotowane pomyślnie.')
    return X, y, le

def train_rnn_model(X, y):
    logger.info('Trenowanie modelu RNN.')
    model = Sequential()
    model.add(Embedding(input_dim=np.max(X) + 1, output_dim=50, input_length=X.shape[1]))
    model.add(LSTM(100))
    model.add(Dense(np.max(y) + 1, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10)
    logger.info('Model RNN wytrenowany pomyślnie.')
    return model

def get_rnn_recommendations(customer_id, model, le, df):
    seq = df[df['customer_id'] == customer_id]['product_id'].values
    encoded_seq = le.transform(seq)
    padded_seq = pad_sequences([encoded_seq], maxlen=model.input_shape[1], padding='pre')
    pred = model.predict(padded_seq)
    recommended_product = le.inverse_transform([np.argmax(pred)])[0]
    return [recommended_product]

# Collaborative Filtering (implicit)
def prepare_cf_data(df):
    logger.info('Przygotowywanie danych dla Collaborative Filtering.')
    customer_product_matrix = df.pivot(index='customer_id', columns='product_id', values='purchase_date').fillna(0)
    customer_product_matrix[customer_product_matrix > 0] = 1
    sparse_matrix = csr_matrix(customer_product_matrix.values)
    logger.info('Dane dla Collaborative Filtering przygotowane pomyślnie.')
    return sparse_matrix, customer_product_matrix

def train_cf_model(sparse_matrix):
    logger.info('Trenowanie modelu Collaborative Filtering.')
    model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
    model.fit(sparse_matrix)
    logger.info('Model Collaborative Filtering wytrenowany pomyślnie.')
    return model

def get_cf_recommendations(customer_id, model, customer_product_matrix, top_n=10):
    customer_index = list(customer_product_matrix.index).index(customer_id)
    recommendations = model.recommend(customer_index, customer_product_matrix, N=top_n)
    product_ids = [customer_product_matrix.columns[i] for i, _ in recommendations]
    return product_ids

def save_recommendations(customer_id, algorithm, recommendations, probability):
    rec_str = ",".join(map(str, recommendations))
    recommendation = Recommendation(customer_id=customer_id, algorithm=algorithm, recommendations=rec_str, probability=probability)
    db.session.add(recommendation)
    db.session.commit()

def calculate_optimal_purchase_interval(df):
    logger.info('Obliczanie optymalnego czasu między zakupami.')
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df = df.sort_values(by=['customer_id', 'purchase_date'])
    df['previous_purchase_date'] = df.groupby('customer_id')['purchase_date'].shift(1)
    df['days_between_purchases'] = (df['purchase_date'] - df['previous_purchase_date']).dt.days
    optimal_intervals = df.groupby('customer_id')['days_between_purchases'].median().reset_index()
    optimal_intervals.columns = ['customer_id', 'optimal_days_between_purchases']
    logger.info('Optymalny czas między zakupami obliczony pomyślnie.')
    return optimal_intervals

def generate_recommendations():
    try:
        df = get_data()
        
        # Obliczanie optymalnego czasu między zakupami
        optimal_intervals = calculate_optimal_purchase_interval(df)
        optimal_intervals.to_sql('optimal_intervals', db.engine, if_exists='replace', index=False)

        # Association Rules
        rules, basket = generate_association_rules(df)
        for customer_id in df['customer_id'].unique():
            assoc_recommendations = get_association_recommendations(customer_id, rules, basket)
            save_recommendations(customer_id, 'association_rules', assoc_recommendations, 0.8)
        
        # RNN
        X, y, le = prepare_rnn_data(df)
        rnn_model = train_rnn_model(X, y)
        for customer_id in df['customer_id'].unique():
            rnn_recommendations = get_rnn_recommendations(customer_id, rnn_model, le, df)
            save_recommendations(customer_id, 'rnn', rnn_recommendations, 0.7)
        
        # Collaborative Filtering
        sparse_matrix, customer_product_matrix = prepare_cf_data(df)
        cf_model = train_cf_model(sparse_matrix)
        for customer_id in df['customer_id'].unique():
            cf_recommendations = get_cf_recommendations(customer_id, cf_model, customer_product_matrix)
            save_recommendations(customer_id, 'collaborative_filtering', cf_recommendations, 0.9)

        logger.info('Rekomendacje wygenerowane pomyślnie.')
    except Exception as e:
        logger.error(f'Błąd podczas generowania rekomendacji: {e}')
