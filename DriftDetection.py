import pandas as pd

import dataset
from config import *


class DriftDetection:
    def __init__(self, df, detector_type, **kwargs):
        self.stream_dates, self.stream_data = dataset.editting(df)
        self.detector = detector_type(**kwargs)
        self.drift_dates = []
        self.drift_cols = []

    def stream_val(self, i, col):
        val = self.stream_data.iloc[i, self.stream_data.columns.get_loc(col)]
        return val

    # def update(self, i, col):
    #
    #     val = self.stream_val(i, col)
    #     self.detector.update(val)

    def detection_mess(self, i, col):
        mess = 'Change detected in data: ' + str(self.stream_val(i, col)) + \
               ' - at index: ' + str(self.stream_dates[i]) + \
               'for column:' + col
        return mess

    def get_drifts(self, i, col, val):

        self.detector.update(val)

        if self.detector.change_detected:
            print(self.detection_mess(i, col))
            self.drift_dates.append(self.stream_dates[i])
            self.drift_cols.append(col)
            self.detector.reset()

    def stream_detection(self):
        for col in self.stream_data.columns:
            for i, val in enumerate(self.stream_data[col]):
        # for i, row in self.stream_data.iterrows():
        #     for col, val in row.items():
                self.get_drifts(i, col, val)
        self.detector.reset()

        drift_df = pd.DataFrame({
            'date':self.drift_dates,
            'feature': self.drift_cols
        })

        return drift_df

    # def elbow(self):

def feature_drifts(drift_df, col):

    return drift_df.loc[drift_df.feature == col, 'date']


