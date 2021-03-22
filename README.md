# Electricity-forecasting

Use time series data to predict operating reserve of electrical. I used an ARIMA model to forecast future values.

<img src="https://github.com/gsdndeer/Electricity-forecasting/blob/main/figures/operating_reserve.jpg" alt="operating reserve from 20200101 to 20210321">

## Usage

``` python app.py --training 20200101_20210321.csv --output submission.csv```


## Date
1. [台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995)
2. [台灣電力公司_本年度每日尖峰備轉容量率](https://data.gov.tw/dataset/25850)
