from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

## Subroutine to compute mutual info classification score for each column in a DataFrame
## call this repeatedly to see if feature engineering has resulted in improved features
def best_features(df, target_name='cmte_pty', seed=1234567):
    """
    Prints the estimated feature importance score of the columns in the data_frame 
    (feature importance here is the mutual information between the target and predictor)
    
    :param df
    :type pandas.DataFrame
    The dataframe with the engineered features.
    
    :param target_name
    :type str
    The name of the column in df that is the label.
    
    :param seed
    :type int
    Seed to ensure the train_test_split results in a repeatable outcome

    :returns None
    """
    # Separate our X and Y data
    data_Y = df[target_name]
    data_X = df.drop([target_name],axis=1)
    print("X shape: ", data_X.shape) 
    print("Y shape: ", data_Y.shape)

    # An 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.20, random_state = seed) #set seed so we can keep running this
    print("X_train = ", X_train.shape, " y_train = ", y_train.shape)
    print("X_test = ", X_test.shape, " y_test = ", y_test.shape)
    
    def mic_wseed(seed):
        def _mutual_info_classif(*args, **kwargs):
            kwargs['random_state'] = seed
            return mutual_info_classif(*args, **kwargs)
        return _mutual_info_classif.__call__

    # prepare input data
    def prepare_inputs(X_train, X_test):
        oe = OrdinalEncoder()
        oe.fit(data_X)
        X_train_enc = oe.transform(X_train)
        X_test_enc = oe.transform(X_test)
        return X_train_enc, X_test_enc

    # prepare target
    def prepare_targets(y_train, y_test):
        le = LabelEncoder()
        le.fit(data_Y)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        return y_train_enc, y_test_enc

    def select_features(X_train, y_train, X_test):
        fs = SelectKBest(score_func=mic_wseed(seed), k='all')
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs

    # prepare input data
    X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)

    # prepare output data
    y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)

    # what are scores for the features
    for i in range(len(fs.scores_)):
        print('Feature %s: %f' % (data_X.columns[i], fs.scores_[i]))

    # plot the scores
    #pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    #pyplot.show()