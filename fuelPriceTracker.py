import pandas as pd

def get_pump_price_data(path, sheet_name='Sheet2'):
    # Load the pump price data
    return pd.read_excel(path)

def get_crude_oil_price_data(path):
    # Load the crude oil price data
    return pd.read_excel(path)

pump = get_pump_price_data('PUMPPRICEPMS.xlsx')
crude_oil = get_crude_oil_price_data('crude_Oilprice.xlsx')

