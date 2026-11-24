import pandas as pd

def get_fuel_price_data(file_path):

    cpd = pd.read_csv(file_path)
    cpd.columns = ["Date","Crude Oil Price"]
    cpd['Date'] = pd.to_datetime(cpd['Date'], format="%m/%d/%Y")
    cpd = cpd.set_index('Date')

    return cpd

d = get_fuel_price_data("fuel_price_data.csv")

for i in (1,2,3,6,12):
    d[f"lag{i} "] = d['Crude Oil Price'].shift(i)
d['"roll3'] = d['Crude Oil Price'].rolling(3).mean()
d['"roll12'] = d['Crude Oil Price'].rolling(12).mean()
d["chg1"] = d['Crude Oil Price'].pct_change()
d["target"]= d['Crude Oil Price'].shift(-1)
d = d.dropna()