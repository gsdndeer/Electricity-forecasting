import argparse
import pandas as pd
import matplotlib.pyplot as pyplot
from pmdarima.arima import auto_arima


def load_csv(csv_name):
    df = pd.read_csv(csv_name, parse_dates=['date'])
    data_train = df[['date','power']]
    data_train = data_train.set_index('date')

    return data_train


def arima(data_train):
    arima_model =  auto_arima(data_train, start_p=2, d=None, start_q=2,
                              max_p=5, max_d=2, max_q=5, start_P=1, 
                              D=None, start_Q=1, max_P=2, max_D=1, 
                              max_Q=2, max_order=5, m=1, seasonal=False, 
                              stationary=True, information_criterion='aic', 
                              alpha=0.05, test='kpss', seasonal_test='ocsb', 
                              stepwise=True, n_jobs=1, start_params=None, trend=None, 
                              method='lbfgs', maxiter=50, offset_test_args=None, 
                              seasonal_test_args=None, suppress_warnings=True, 
                              error_action='trace', trace=False, random=False, 
                              random_state=None, n_fits=10, out_of_sample_size=0, 
                              scoring='mse', scoring_args=None, with_intercept='auto')
    # predict
    data_predict = arima_model.predict(n_periods=7)
    date = {'date' : ['20210323', '20210324', '20210325', '20210326', '20210327', '20210328', '20210329'],
            'operating_reserve(MW)' : data_predict}
    prediction = pd.DataFrame(date)

    return prediction


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training', 
                        default='20200101_20210318.csv',
                        help= 'input training data file name')
    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # load data
    data_train = load_csv(args.training)
    # predict
    data_predict = arima(data_train)
    # save date
    data_predict.to_csv(args.output, index=0)