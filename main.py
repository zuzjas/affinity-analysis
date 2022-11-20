import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def load_data():
    data = pd.read_csv('Sales_September_2019.csv')
    data = data.dropna()

    return data


def affinity_analysis(data):
    # stworzenie koszyka produktow
    basket_data = data.groupby(['Order ID', 'Product'])['Quantity Ordered'].sum().unstack().reset_index().fillna(
        0).set_index('Order ID')

    # zakodowanie danych (0-produkt niekupiony, 1-produkt kupiony)
    basket_data = basket_data.applymap(encode_units)

    # odrzucenie transakcji, gdzie klient kupił 1 typ produktu
    basket_data = basket_data[(basket_data > 0).sum(axis=1) >= 2]

    # algorytm eksploracji danych - Apriori algorithm
    frequent_items = apriori(basket_data, min_support=0.03, use_colnames=True).sort_values('support',
                                                                                           ascending=False).reset_index(
        drop=True)
    frequent_items['length'] = frequent_items['itemsets'].apply(lambda x: len(x))

    # reguły asocjacyjne
    a_rules = association_rules(frequent_items, metric="lift", min_threshold=1).sort_values('lift',
                                                                                            ascending=False).reset_index(
        drop=True)

    return a_rules


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


if __name__ == '__main__':
    data = load_data()
    a_rules = affinity_analysis(data)
    a_rules.to_csv('results.csv')
