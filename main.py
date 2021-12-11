
from DriftDetection import DriftDetection
import dataset

from river.drift import ADWIN
from config import *

df = dataset.get_data(DATA_DIR, DATASET)

adwin_det = DriftDetection(df, ADWIN, delta=0.001)
dates, cols = adwin_det.stream_detection()

print('done')

