import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from utils import haversineKaggle

           
DATA_DIR = '../data'

t0 = time.time()
for i in range(1, 5):
    filename = 'train_pset_' + str(i)
    print('reading training data from %s ...' % filename)
 
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    d1 = haversineKaggle(df[['xs', 'ys']].values, df[['xe', 'ye']].values)

    # create training set
    y = np.log((df['len'] - 1)*15)
    # remove non-predictive features
    df.drop(['CALL_TYPE', 'TAXI_ID', 'xe', 'ye', 'len'], axis=1, inplace=True)
    X = np.array(df, dtype=np.float)

    # clean data by removing long distance tracks
    th1 = np.percentile(d1, [99.9])
    X = X[(d1<th1), :]
    y = y[(d1<th1)]
                                                   
    print('training a random forest regressor ...')
    # Initialize the famous Random Forest Regressor from scikit-learn
    clf = RandomForestRegressor(n_estimators=200, n_jobs=3, random_state=21)
    clf.fit(X, y)

    print('predicting test data ...')
    df = pd.read_csv(os.path.join(DATA_DIR, filename.replace('train', 'test')))
    ids = df['TRIP_ID']
    
    df = df.drop(['TRIP_ID', 'CALL_TYPE', 'TAXI_ID'], axis = 1)
    X_tst = np.array(df, dtype=np.float)
    y_pred = clf.predict(X_tst)

    # create submission file
    submission = pd.DataFrame(ids, columns=['TRIP_ID'])
    filename = filename.replace('train_set', 'my_submission')
    submission['TRAVEL_TIME'] = np.exp(y_pred)
    submission.to_csv(filename, index = False)

print('Done in %.1f sec.' % (time.time() - t0))


print('reading general model predictions ...')
df = pd.read_csv('../data/test_set_1.csv')

idx = np.argsort(df['length'])
for i in range(1, 5): 
    submission_file = 'my_submission_' + str(i) + '.csv'
    tmp = pd.read_csv(submission_file)
    df['F%i' % i] = tmp['TRAVEL_TIME']

print('creating submission 1 ...')
df['TRAVEL_TIME'] = df['F4']
idx = np.where(df['length']<=15)[0]
df.loc[idx, 'TRAVEL_TIME'] = np.mean(df.loc[idx, ['F1','F2','F3','F4']], axis=1)

submission = df[['TRIP_ID','TRAVEL_TIME']]
submission.to_csv('final_submission_1.csv', index = False)