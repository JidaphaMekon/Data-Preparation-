# %% [markdown]
# Agriculture and food industry_5y_data.xlsx

# %%
%pip install TA-Lib


# %%
# นำเข้าไลบรารีที่จำเป็น
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# อ่านข้อมูลจาก Excel แทน yfinance
file_path = 'Agriculture and food industry_5y_data.xlsx'  # 🔁 เปลี่ยนชื่อไฟล์ตามของจริง
df = pd.read_excel(file_path)

# ตรวจสอบว่าคอลัมน์ 'Date' อยู่ใน index หรือเปล่า
if df.index.name == 'Date':
    df = df.reset_index()

# กรองคอลัมน์ให้มีเฉพาะที่ต้องการ (ถ้ามีมากกว่านี้ในไฟล์)
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# แปลงคอลัมน์ Date ให้อยู่ในรูป datetime (ถ้ายังไม่ได้แปลง)
df['Date'] = pd.to_datetime(df['Date'])

# ดูตัวอย่างข้อมูล
print(df.head())


# %%
import pandas as pd
import talib as ta

# 1. อ่านข้อมูลจากทุกชีตในไฟล์ Excel
file_path = 'Agriculture and food industry_5y_data.xlsx'
sheets_dict = pd.read_excel(file_path, sheet_name=None)

period = 14
processed_dict = {}

def rsi_signal(rsi):
    if rsi >= 70:
        return "SELL"
    elif rsi <= 30:
        return "BUY"
    else:
        return ""

def signal_to_binary(signal):
    if signal == "BUY":
        return 1
    else:
        return 0

for sheet_name, df in sheets_dict.items():
    if df.index.name == 'Date':
        df = df.reset_index()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    df['EMA25'] = ta.EMA(df['Close'], timeperiod=25)
    df['EMA75'] = ta.EMA(df['Close'], timeperiod=75)
    df['EMA200'] = ta.EMA(df['Close'], timeperiod=200)
    df['RSI'] = ta.RSI(df['Close'], timeperiod=period)

    macd, macdsignal, macdhist = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macdsignal
    df['MACD_hist'] = macdhist

    df['Change'] = df['Close'].diff()
    df['Gain'] = df['Change'].apply(lambda x: x if x > 0 else 0)
    df['Loss'] = df['Change'].apply(lambda x: -x if x < 0 else 0)

    df['Avg Gain'] = df['Gain'].rolling(window=period).mean()
    df['Avg Loss'] = df['Loss'].rolling(window=period).mean()

    df['RS'] = df.apply(lambda row: row['Avg Gain'] / row['Avg Loss'] if row['Avg Loss'] != 0 else 100, axis=1)

    df['RSI Signal'] = df['RSI'].apply(rsi_signal)

    # เพิ่มคอลัมน์ status แบบ binary target
    df['status'] = df['RSI Signal'].apply(signal_to_binary)

    processed_dict[sheet_name] = df

    # แสดงค่า unique ของ status ในแต่ละชีต
    print(f"Sheet: {sheet_name} — unique values in status: {df['status'].unique()}")

# Export ทุกชีตกลับไปยัง Excel
output_path = 'หุ้น_processed.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for sheet, df in processed_dict.items():
        df.to_excel(writer, sheet_name=sheet, index=False)



