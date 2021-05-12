import functools
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from constants import *

pd.options.mode.chained_assignment = None


def strcolor(colorscale, temp_color):
  max_color = MAX_RANGE
  min_color = MIN_RANGE
  if temp_color > max_color:
      return colorscale[-1][1]
  if temp_color < min_color:
      return colorscale[0][1]
  color_idx = int(((temp_color - min_color) / (max_color - min_color)) * (len(colorscale) - 1))
  return colorscale[color_idx][1]

def split_data(X, y, scale=True):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

  # Заменяем нулевые значения
  for feature in X.columns:
    X_train[feature].fillna(X_train[feature].mean(), inplace=True)
    X_test[feature].fillna(X_test[feature].mean(), inplace=True)

  # Скейлинг нужен для работы некоторых байесовских методов
  if scale:
    features_to_scale = ['MinTemp', 'MaxTemp']

    for feature in features_to_scale:
      scaler = MinMaxScaler()
      X_test[feature] = scaler.fit_transform(X_test[feature].values.reshape(-1, 1))
      X_train[feature] = scaler.fit_transform(X_train[feature].values.reshape(-1, 1))
  return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

@functools.cache
def load_data():
    df = pd.read_csv('data/weatherAUS.csv', sep=',', parse_dates=['Date'])
    df['RainTomorrow'] = df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['RainToday'] = df['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)
    df = df.drop('RISK_MM', axis=1)

    month_col = df['Date'].dt.month
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    month_to_season = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}
    for i, season in enumerate(seasons):
        df[season] = month_col.apply(lambda month: 1 if (month_to_season[month] == i + 1) else 0)
    df = df.drop('Date', axis=1)

    df = df.drop('Location', axis=1)
    # # Разбиваем категориальный признак на столбцы
    # df = pd.get_dummies(df, columns=['Location'])
    # Убираем коррелирующие признаки
    df = df.drop(['Temp9am', 'Temp3pm', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Pressure9am'], axis=1)

    y = df['RainTomorrow']
    X = df[df.columns.drop('RainTomorrow')]
    return split_data(X, y, scale=True)
