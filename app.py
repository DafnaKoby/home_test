
import streamlit as st
import plotly.express as px

import dataset
import plots

from config import *
from DriftDetection import DriftDetection, feature_drifts
from river.drift import ADWIN


def main():
    df = dataset.get_data(DATA_DIR, DATASET)
    st.header('Data Science Home Assignment')
    st.subheader('Dafna Koby')

    st.markdown('For the code generating this app, click [here](https://github.com/DafnaKoby/home_test)')

    st.markdown('## Data Exploration')
    st.markdown('In the table below...')

    st.write(df.head(100))

    hist_cols = st.selectbox('Features:', NUMERIC_FEATURES + CAT_FEATURES, key='hist_feat')
    histograms = plots.plot_hist(df, hist_cols)
    st.plotly_chart(histograms)

    st.markdown('### Drift')

    fig = plots.plot_line(df, 'score', p_title='Average Score')
    st.plotly_chart(fig)

    st.markdown('''
    The plot above indicates a possible drift has started following February 19th. 
    While prior to it, the average score is stationary, and ranging from 0.45 to 0.47, following February 19th,
    the average score sharply climbs to approximately 0.52.\n
    We should therefore look for any drift occurring in that period, in one or more features.     
    ''')

    st.markdown('## Task 1')

    st.markdown('''
        The plots bellow shoe the average daily value of the numeric features, and the daily count of the 
        categorical features. Observing the trends over time, we notice two features exhibit sharp changes 
        starting February 19th, which is also the date the model possibly started to deteriorate:
    ''')
    st.markdown('''
            - **external_email_score** - starting February 19th, there is a jump from approximately 2.7, to 
            5.8 on February 26th.
            - **avs_match** - also on February 19th, there is a steep decline in *country* type match, 
            simultaneously with a rise in *None* match.
            - **ip_score** - similarly, on February 19, begins to sharply climb. 
    ''')

    cols = st.selectbox('Features:', NUMERIC_FEATURES + CAT_FEATURES, key='drift_feat')

    if cols in(NUMERIC_FEATURES):
        fig = plots.plot_line(df, cols)
    if cols in(CAT_FEATURES):
        fig = plots.plot_line_cat(df, cols)
    st.plotly_chart(fig)

    st.markdown('## Task 2')
    st.markdown('### Drift Detection Methods')
    st.markdown('''
    Before presenting the chosen model and its results, the next section describes the pros and cons faced when 
    choosing a drift detection method. Most of the problems can be attributed to the trade-off between reacting quickly
    to changes and having few false alarms.''')
    st.markdown('''
        - **Statistical methods** - such as Kolmogorov-Smirnov test, Jensen-Shannon Divergence, and Kullback–Leibler
         Divergence. They are used to compare the difference between distributions: We test a hypothesis of this sort by 
         drawing a random sample from the population in question and calculating an appropriate statistic on its items. 
         If, in doing so, we obtain a value of the statistic that would occur rarely when the hypothesis is true, 
         we would have reason to reject the hypothesis.\n
          On the one hand, we would like to have long windows so that the 
         estimates on each window are more robust, but, on the other hand, short windows to detect a change as soon as it happens. 
         
        - **CUMSUM and Page-Hinckley** - both methods are designed to give an alarm when the mean of the
         input data significantly deviates from its previous value. They are sensitive to the parameter values, 
         resulting in a tradeoff between false alarms and detecting true drifts. 
        
        - **ADWIN (ADaptive sliding WINdow)** - It aims at solving some of the problems in the change estimation and 
        detection methods described before. These can be attributed to the trade-off between reacting quickly 
        to changes and having few false alarms. The idea of ADWIN is to start from time window W and dynamically grow 
        the window W when there is no apparent change in the context, and shrink it when a change is detected.
        
        - **Trees** - To identify a drift, train a tree on the data and add prediction timestamp as one of the features.
        we can know how the time affects the data and at which point. Moreover, we can look at the split created by the
        timestamp and we can see the difference between the concepts before and after the split. This method suffers 
        from similar issues to those of statistical methods, as a time window needs to be chosen.
        
        - ** Model-Based Approach ** - We need to label our data which has been used to build the current model in 
        production as 0 and the real-time data gets labeled as 1. If the model gives high accuracy, it means that it 
        can easily discriminate between the two sets of data. Thus, we could conclude that a covariate shift has 
        occurred and the model will need to be recalibrated. The disadvantage of this model is that every time new input 
        data is made available, the training and testing process needs to be repeated which can become computationally
        expensive. 
    ''')

    st.markdown('''Since, as specified in the task, drifts can occur at different times, features, and intensities
    and we are interested in detecting the drift as early as possible, the **ADWIN** algorithm is chosen.
    ''')

    st.markdown('### Feature Engineering')

    st.markdown('''
        - Each categorical variable is transformed into OHE.
        - In order to alleviate the noise in the data, orders are binned in groups of 100 samples.     
        For numerical features, the average of the batch is calculated, and for categorical, the count of
         each value.''')


    st.markdown('The resulting dataset is below:')

    adwin_det = DriftDetection(df, ADWIN, delta=0.001)

    st.write(adwin_det.stream_data.head(100))

    st.markdown('### Training Dataset')

    st.markdown(''' 
    Following are plots of the features detected to exhibit drifting by the model. Dates on which drift was 
    detected are in dashed, vertical line. 
    ''')

    drifts = adwin_det.stream_detection()

    cols = st.selectbox('Features:', drifts['feature'].unique(), key='drift_res')

    if cols in(NUMERIC_FEATURES):
        fig = plots.plot_line(df, cols)
    else:
        ohe_cat = dataset.ohe(df, CAT_FEATURES)
        fig = plots.plot_line_oh(ohe_cat, cols)

    f_drifts = feature_drifts(drifts, cols)

    fig = plots.plot_drift(fig, f_drifts)
    st.plotly_chart(fig)

    st.markdown('''
        While the model does locate the drift in **external_email_score** and **avs_match**, it appears to be 
        overly sensitive, and flags other instances as well, and among other features. More specifically, 
        there seems to be problem when facing early stream data. This can be attributed to the window being too small
         due to the lack of previous data.   
    ''')

    st.markdown('### Test Dataset')

    st.markdown('''Before analysing the results of the model using the test set, observing the scores of the
    model over March--April, there are two dips: The first from April 23rd to May 3rd, and the second from
    Mat 23rd to the end of the data, on May 31st. 
    ''')

    test_data = dataset.get_data(DATA_DIR, TEST_DATASET)

    fig = plots.plot_line(test_data, 'score', p_title='Average Score')
    st.plotly_chart(fig)

    st.markdown(''' 
      Following are plots of the features detected to exhibit drifting in the test dataset.
        Dates on which drift was detected are in dashed, vertical line, no line implies no drift was detected. 
      ''')

    adwin_test = DriftDetection(test_data, ADWIN, delta=0.001)

    test_drifts = adwin_test.stream_detection()

    cols = st.selectbox('Features:', adwin_test.stream_data.columns, key='drift_test')

    if cols in (NUMERIC_FEATURES):
        fig = plots.plot_line(test_data, cols)
    else:
        ohe_cat = dataset.ohe(test_data, CAT_FEATURES)
        fig = plots.plot_line_oh(ohe_cat, cols)

    if test_drifts['feature'].str.contains(cols).any():
        f_test_drifts = feature_drifts(test_drifts, cols)
        fig = plots.plot_drift(fig, f_test_drifts)

    st.plotly_chart(fig)

    st.markdown('''
     In the test dataset, the feature exhibiting drift is **payment_method**. Specifically, the number of credit
     card orders dips twice, simultaneously to increases in the number of Stripe orders.
     The dates of these changes correspond to the deterioration scores predicted by the model, indicating these 
     are the variables with drift.\n
     While the ADWIN model detects these changes, as with the training set, the model is highly sensitive to 
     changes in the initial period, and reports false positives for features with no drift.  
     ''')

    st.markdown('### Further Work')
    st.markdown('''
    The results of the model chosen indicate more work is needed in order to for it to be helpful in detecting
    data drift.\n
    More concretely:
    ''')
    st.markdown('''
        - The confidence variable, delta, can be chosen more carefully, either by simulations, or by validation
        datasets.
        - Similarly, the batch size can be chosen by validation in order to handle noisy data better.
        - Alternatively to aggregating by batch, more complex smoothing methods can be employed.  
    ''')

    st.markdown('## Task 3')
    st.markdown("![Task 3](https://raw.githubusercontent.com/DafnaKoby/riskified_home_test/master/2021-12-11-task3.PNG)")

    st.markdown('''
        At the offline stage, current data is collected to create a historical dataset in the DWH.
        The historical dataset is then used to tune the hyperparameters of the drift detector.
        At the online stage, the tuned detector obtained from the offline learning is used to process the
         data streams. If concept drift is detected in the new data streams, users are alarmed, and the detector will then be
         retrained on the new concept data samples collected.
    ''')


    st.markdown('## References')
    st.markdown('''
    [Machine Learning for Data Streams: with Practical Examples in MOA](https://www.cms.waikato.ac.nz/~abifet/book/chapter_5.html)\n
    [8 Concept Drift Detection Methods](https://www.aporia.com/blog/concept-drift-detection-methods/)
    ''')


if __name__ == '__main__':
    main()
