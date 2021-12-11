
DATA_DIR = 'https://raw.githubusercontent.com/DafnaKoby/riskified_home_test/master/datasets'
DATASET = 'riskified_home_task_data.csv'
TEST_DATASET = 'riskified_home_task_test_data.csv'

NUMERIC_FEATURES = [
    'price',
    'ip_score',
    'external_email_score',
    'history_count',
    'bill_ship_distance',
    'address_score'
]
CAT_FEATURES = [
    'merchant',
    'avs_match',
    'payment_method'
]

TIME_COL = 'created_at'

BATCH_SIZE = 100
