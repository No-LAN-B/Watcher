#training time

feature_columns = [
    'Open', 
    'High', 
    'Low', 
    # Initially exclude as model could learn trivial results leading to problems'Close',
    'Volume',
    'return',
    'sma_10',
    'vol_10',
    'Close_Lag_1'
    #RSI MACD etc could be added here
]

X = df[feature_columns]
Y = df['Target_Price']