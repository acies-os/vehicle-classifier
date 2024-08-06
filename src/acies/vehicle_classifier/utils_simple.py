import numpy as np
import pandas as pd
import os
from scipy import signal
import numpy as np
import itertools
import pandas as pd
import os
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.model_selection import train_test_split


def filter_samples_for_test(samples):
    # get data from 50 unique ids for testing for fast processing
    samples = samples[samples['id'].isin(samples['id'].unique()[:50])]
    return samples


def extract_features(samples):
    
    TEST = False
    if TEST:
        samples = filter_samples_for_test(samples)
        
    # Remove the timestamp_seconds column
    if 'timestamp_seconds' in samples.columns:
        samples = samples.drop(columns=['timestamp_seconds'])
    
    def apply_features(x, column):
        if column == 'samples_acoustic':
            pass
        elif column == 'samples_seismic':
            # downsample seismic data to 100 from 1000
            x = x[::10]
        return pd.Series(applyAndReturnAllFeatures(x[column].values))
    
    # Extract acoustic features
    features_acoustic = (samples.groupby('id')
                         .apply(lambda x: apply_features(x, 'samples_acoustic'))
                         .reset_index())
    features_acoustic.columns = ['id'] + [f'acoustic_{col}' for col in features_acoustic.columns[1:]]
    
    # Extract seismic features
    features_seismic = (samples.groupby('id')
                        .apply(lambda x: apply_features(x, 'samples_seismic'))
                        .reset_index())
    features_seismic.columns = ['id'] + [f'seismic_{col}' for col in features_seismic.columns[1:]]
    
    # Expand list and object columns
    def expand_list_columns(df):
        list_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, (list, object)) and hasattr(x, "__iter__")).any()]
        # remove id label and formation from list cols
        list_cols = [col for col in list_cols if not col in ['id', 'label', 'formation']]
        
        for col in list_cols:
            expanded = pd.DataFrame(df[col].tolist(), index=df.index)
            expanded.columns = [f'{col}_{i}' for i in range(expanded.shape[1])]
            df = df.drop(columns=[col]).join(expanded)
        return df
    
    features_acoustic = expand_list_columns(features_acoustic)
    features_seismic = expand_list_columns(features_seismic)
    
    # Merge the two feature dataframes
    features = pd.merge(features_acoustic, features_seismic, on='id')
    
    # Add id, label, and formation back to the dataframe
    metadata = samples[['id', 'label', 'formation']].drop_duplicates().reset_index(drop=True)
    features = pd.merge(metadata, features, on='id')
    
    return features


def fix_target_with_pedestrian_cases(all_features):
    # if label is all_features[all_features['label']=='[6, 1]'], then convert to label 6 and change the formation column to 0    
    # find rows with label 6, 1
    pedestrian_rows = all_features[all_features['label']=='[6, 1]']
    if pedestrian_rows.shape[0] == 0:
        return all_features
    pedestrian_rows['label'] = 6
    pedestrian_rows['formation'] = 0
    pedestrian_rows['id'] = pedestrian_rows['id'].apply(lambda x: (f"6_{x[0].split('_')[1]}", x[1]))
    all_features[all_features['label']=='[6, 1]'] = pedestrian_rows
    return all_features




def train_model(all_features, model_version, target='label'):
    BASE_DATA_PATH = "/data/kara4/2023-graces-quarters"
    MODELS_PATH = os.path.join(BASE_DATA_PATH, "models")

    # easy fix for warthog pedestrian cases
    all_features = fix_target_with_pedestrian_cases(all_features)
    
    if target == 'label':
        # remove labels with multiple targets
        all_features = all_features[all_features['formation'] == 0]
        # remove formation column
        del all_features['formation']

        # id is in format of ({label}_{run}, timestamp)
        unique_ids = all_features['id'].apply(lambda x: x[0]).unique()
        
        # Split the unique_ids into labels and runs
        unique_ids_df = pd.DataFrame(unique_ids, columns=['id'])
        unique_ids_df['label'] = unique_ids_df['id'].apply(lambda x: x.split('_')[0])
        unique_ids_df['run'] = unique_ids_df['id'].apply(lambda x: x.split('_')[1])
        
        # Create directory for model version
        model_version_path = os.path.join(MODELS_PATH, f"{model_version}_label")
        os.makedirs(model_version_path, exist_ok=True)
        
        for current_run in range(5):
            # Create train-test split ensuring all labels are in both sets
            train_ids, test_ids = train_test_split(unique_ids_df, test_size=0.1, stratify=unique_ids_df['label'], random_state=current_run)
            
            # Extract the list of ids for training and testing
            train_ids_list = train_ids['id'].tolist()
            test_ids_list = test_ids['id'].tolist()
            
            # Split the all_features based on train_ids and test_ids
            train_features = all_features[all_features['id'].apply(lambda x: x[0]).isin(train_ids_list)]
            test_features = all_features[all_features['id'].apply(lambda x: x[0]).isin(test_ids_list)]
            
            # Now you can proceed with training your model using train_features and testing using test_features
            print(f"Run {current_run+1}:")
            print("Shape of all features: ", all_features.shape)
            print("Shape of train features: ", train_features.shape)
            print("Shape of test features: ", test_features.shape)
            
            # drop id column from train and test features
            try:
                train_features = train_features.drop(columns=["id"])
                test_features = test_features.drop(columns=["id"])
            except KeyError:
                print("id column does not exist")
            
            # Ensure label encoding is consistent
            labels = sorted(all_features['label'].unique())
            label_to_index = {label: idx for idx, label in enumerate(labels)}
            train_features['label'] = train_features['label'].map(label_to_index)
            test_features['label'] = test_features['label'].map(label_to_index)
            
            # Train the model
            print("Training the model...")
            model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=200)
            model.fit(train_features.drop(columns=["label"]), train_features["label"])

            # Save the model to pkl file
            model_name = f"model_{current_run+1}"
            pickle.dump(model, open(os.path.join(model_version_path, model_name + ".pkl"), "wb"))

            # Predict the test data
            test_labels = test_features["label"]
            y_pred = model.predict(test_features.drop(columns=["label"]))

            # Calculate accuracy, precision, recall, f1 score
            accuracy = accuracy_score(test_labels, y_pred)
            precision = precision_score(test_labels, y_pred, average="macro")
            recall = recall_score(test_labels, y_pred, average="macro")
            f1 = f1_score(test_labels, y_pred, average="macro")
            print("Accuracy: ", accuracy)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("F1: ", f1)

            # Save these results to a csv file
            results = pd.DataFrame({
                "metric": ["accuracy", "precision", "recall", "f1"],
                "value": [accuracy, precision, recall, f1],
                "run": current_run + 1
            })
            results.to_csv(os.path.join(model_version_path, model_name + "_results.csv"), index=False)

            try:
                # Save a confusion matrix
                cm = confusion_matrix(test_labels, y_pred)
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                cm_df.to_csv(os.path.join(model_version_path, model_name + "_confusion_matrix.csv"))
            except Exception as e:
                print(f"Could not save confusion matrix: {e}")

            # Print the confusion matrix
            try:
                print(cm_df)
            except Exception as e:
                print(f"Could not print confusion matrix: {e}")

    elif target == 'formation':
        # just take the formation 0 (single) or 2 (trailing multiple) cases and train the model
        all_features = all_features[all_features['formation'].isin([0, 2])]
        # remove label column
        del all_features['label']
        
        # id is in format of ({label}_{run}, timestamp)
        unique_ids = all_features['id'].apply(lambda x: x[0]).unique()
        
        # Split the unique_ids into labels and runs
        unique_ids_df = pd.DataFrame(unique_ids, columns=['id'])
        unique_ids_df['label'] = unique_ids_df['id'].apply(lambda x: x.split('_')[0])
        unique_ids_df['run'] = unique_ids_df['id'].apply(lambda x: x.split('_')[1])
        
        # Create directory for model version
        model_version_path = os.path.join(MODELS_PATH, f"{model_version}_formation")
        os.makedirs(model_version_path, exist_ok=True)
        
        for current_run in range(5):
            # Create train-test split ensuring all labels are in both sets
            train_ids, test_ids = train_test_split(unique_ids_df, test_size=0.1, stratify=unique_ids_df['label'], random_state=current_run)
            
            # Extract the list of ids for training and testing
            train_ids_list = train_ids['id'].tolist()
            test_ids_list = test_ids['id'].tolist()
            
            # Split the all_features based on train_ids and test_ids
            train_features = all_features[all_features['id'].apply(lambda x: x[0]).isin(train_ids_list)]
            test_features = all_features[all_features['id'].apply(lambda x: x[0]).isin(test_ids_list)]
            
            # Now you can proceed with training your model using train_features and testing using test_features
            print(f"Run {current_run+1}:")
            print("Shape of all features: ", all_features.shape)
            print("Shape of train features: ", train_features.shape)
            print("Shape of test features: ", test_features.shape)
            
            # drop id column from train and test features
            try:
                train_features = train_features.drop(columns=["id"])
                test_features = test_features.drop(columns=["id"])
            except KeyError:
                print("id column does not exist")
                
            # Ensure formation encoding is consistent
            formations = sorted(all_features['formation'].unique())
            formation_to_index = {formation: idx for idx, formation in enumerate(formations)}
            train_features['formation'] = train_features['formation'].map(formation_to_index)
            test_features['formation'] = test_features['formation'].map(formation_to_index)
            
            # Train the model
            print("Training the model...")
            model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=200)
            model.fit(train_features.drop(columns=["formation"]), train_features["formation"])

            # Save the model to pkl file
            model_name = f"model_{current_run+1}"
            pickle.dump(model, open(os.path.join(model_version_path, model_name + ".pkl"), "wb"))

            # Predict the test data
            test_labels = test_features["formation"]
            y_pred = model.predict(test_features.drop(columns=["formation"]))

            # Calculate accuracy, precision, recall, f1 score
            accuracy = accuracy_score(test_labels, y_pred)
            precision = precision_score(test_labels, y_pred, average="macro")
            recall = recall_score(test_labels, y_pred, average="macro")
            f1 = f1_score(test_labels, y_pred, average="macro")
            print("Accuracy: ", accuracy)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("F1: ", f1)

            # Save these results to a csv file
            results = pd.DataFrame({
                "metric": ["accuracy", "precision", "recall", "f1"],
                "value": [accuracy, precision, recall, f1],
                "run": current_run + 1
            })
            results.to_csv(os.path.join(model_version_path, model_name + "_results.csv"), index=False)
            
            try:
                # Save a confusion matrix
                cm = confusion_matrix(test_labels, y_pred)
                cm_df = pd.DataFrame(cm, index=formations, columns=formations)
                cm_df.to_csv(os.path.join(model_version_path, model_name + "_confusion_matrix.csv"))
            except Exception as e:
                print(f"Could not save confusion matrix: {e}")

            # Print the confusion matrix
            try:
                print(cm_df)
            except Exception as e:
                print(f"Could not print confusion matrix: {e}")
                

##### ADDED FEATURES #####

# TODO: Ratio and energy features for localization
list_of_features = ["abs_energy", 
                    "absolute_maximum",
                    "count_above_mean",
                    "first_location_of_maximum",
                    "last_location_of_maximum",
                    "longest_strike_above_mean",
                    "mean_change",
                    "variation_coefficient",
                    "welch_spectrogram", # more rectangular window 'rect', we are losing information
                    "fourier_entropy",
                    "fft_aggregated"
                    ]

# nulling out 35 -75- 105 around peaks, add baseband noise isntead
# spectral subtraction on fog machine with alpha=2 parameter.

# TODO: Add more features:
# 1. Aggregate FFT statistics

def applyAndReturnAllFeatures(x):
    """
    Applies all features in the list of features and returns a dictionary with the feature names as keys and the feature values as values.

    :param x: the time series to calculate the features of
    :type x: numpy.ndarray
    :return: the dictionary with the feature names as keys and the feature values as values
    :return type: dict
    """
    return {f: globals()[f](x) for f in list_of_features}


def abs_energy(x):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\\ldots, n} x_i^2

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.dot(x, x)

def welch_spectrogram(x):
    """
    Generates the welch spectrogram of the time series.
    """
    # remove mean from x
    # x = x - np.mean(x) # TODO: try mean removal, duplicate info
    
    # sample len is len(x) for our case
    fs = len(x)
    nperseg = min(len(x), 256)
    f, Pxx_den = signal.welch(x, fs = fs,nperseg=nperseg)
    return Pxx_den

def absolute_maximum(x):
    """
    Calculates the highest absolute value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.max(np.absolute(x)) if len(x) > 0 else np.NaN

def count_above_mean(x):
    """
    Returns the number of values in x that are higher than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    m = np.mean(x)
    return np.where(x > m)[0].size

def first_location_of_maximum(x):
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    #if not isinstance(x, (np.ndarray, pd.Series)):
    #    x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN

def last_location_of_maximum(x):
    """
    Returns the relative last location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN

def longest_strike_above_mean(x):
    """
    Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    #if not isinstance(x, (np.ndarray, pd.Series)):
    #    x = np.asarray(x)
    return np.max(_get_length_sequences_where(x > np.mean(x))) if x.size > 0 else 0

def mean_abs_change(x):
    """
    Returns the mean over the absolute differences between subsequent time series values which is

    .. math::

        \\frac{1}{n} \\sum_{i=1,\\ldots, n-1} | x_{i+1} - x_{i}|


    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.mean(np.abs(np.diff(x)))

def mean_change(x):
    """
    Average over time series differences.

    Returns the mean over the differences between subsequent time series values which is

    .. math::

        \\frac{1}{n-1} \\sum_{i=1,\\ldots, n-1}  x_{i+1} - x_{i} = \\frac{1}{n-1} (x_{n} - x_{1})

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = np.asarray(x)
    return (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else np.NaN

def variation_coefficient(x):
    """
    Returns the variation coefficient (standard error / mean, give relative value of variation around mean) of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan

def _get_length_sequences_where(x):
    """
    This method calculates the length of all sub-sequences where the array x is either True or 1.

    Examples
    --------
    >>> x = [0,1,0,0,1,1,1,0,0,1,0,1,1]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,True,True,True,0,0,True,0,True,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,1,True,1,0,0,True,0,1,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    :param x: An iterable containing only 1, True, 0 and False values
    :return: A list with the length of all sub-sequences where the array is either True or False. If no ones or Trues
    contained, the list [0] is returned.
    """
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]
    
def binned_entropy(x, max_bins=10):
    """
    First bins the values of x into max_bins equidistant bins.
    Then calculates the value of

    .. math::

        - \\sum_{k=0}^{min(max\\_bins, len(x))} p_k log(p_k) \\cdot \\mathbf{1}_{(p_k > 0)}

    where :math:`p_k` is the percentage of samples in bin :math:`k`.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param max_bins: the maximal number of bins
    :type max_bins: int
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)

    # nan makes no sense here
    if np.isnan(x).any():
        return np.nan

    hist, bin_edges = np.histogram(x, bins=max_bins)
    probs = hist / x.size
    probs[probs == 0] = 1.0
    return -np.sum(probs * np.log(probs))

def fourier_entropy(x, bins=10):
    """
    Calculate the binned entropy of the power spectral density of the time series
    (using the welch method).

    Ref: https://hackaday.io/project/707-complexity-of-a-time-series/details
    Ref: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html

    """
    _, pxx = signal.welch(x, nperseg=min(len(x), 256))
    if np.max(pxx) == 0:
        print("Warning: max power spectral density is 0, returning 0")
        return 0  # or return np.nan or some other value that makes sense in your context
    
    return binned_entropy(pxx / np.max(pxx), bins)

def fft_aggregated(x):
    """
    Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"aggtype": s} where s str and in ["centroid", "variance",
        "skew", "kurtosis"]
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    # param = ["centroid", "variance", "skew", "kurtosis"]
    param = [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]]
    
    def get_moment(y, moment):
        """
        Returns the (non centered) moment of the distribution y:
        E[y**moment] = \\sum_i[index(y_i)^moment * y_i] / \\sum_i[y_i]

        :param y: the discrete distribution from which one wants to calculate the moment
        :type y: pandas.Series or np.array
        :param moment: the moment one wants to calcalate (choose 1,2,3, ... )
        :type moment: int
        :return: the moment requested
        :return type: float
        """
        return y.dot(np.arange(len(y), dtype=float)**moment) / y.sum()

    def get_centroid(y):
        """
        :param y: the discrete distribution from which one wants to calculate the centroid
        :type y: pandas.Series or np.array
        :return: the centroid of distribution y (aka distribution mean, first moment)
        :return type: float
        """
        return get_moment(y, 1)

    def get_variance(y):
        """
        :param y: the discrete distribution from which one wants to calculate the variance
        :type y: pandas.Series or np.array
        :return: the variance of distribution y
        :return type: float
        """
        return get_moment(y, 2) - get_centroid(y) ** 2

    def get_skew(y):
        """
        Calculates the skew as the third standardized moment.
        Ref: https://en.wikipedia.org/wiki/Skewness#Definition

        :param y: the discrete distribution from which one wants to calculate the skew
        :type y: pandas.Series or np.array
        :return: the skew of distribution y
        :return type: float
        """

        variance = get_variance(y)
        # In the limit of a dirac delta, skew should be 0 and variance 0.  However, in the discrete limit,
        # the skew blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if variance < 0.5:
            return np.nan
        else:
            return (
                get_moment(y, 3) - 3 * get_centroid(y) * variance - get_centroid(y)**3
            ) / get_variance(y)**(1.5)

    def get_kurtosis(y):
        """
        Calculates the kurtosis as the fourth standardized moment.
        Ref: https://en.wikipedia.org/wiki/Kurtosis#Pearson_moments

        :param y: the discrete distribution from which one wants to calculate the kurtosis
        :type y: pandas.Series or np.array
        :return: the kurtosis of distribution y
        :return type: float
        """

        variance = get_variance(y)
        # In the limit of a dirac delta, kurtosis should be 3 and variance 0.  However, in the discrete limit,
        # the kurtosis blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if variance < 0.5:
            return np.nan
        else:
            return (
                get_moment(y, 4) - 4 * get_centroid(y) * get_moment(y, 3)
                + 6 * get_moment(y, 2) * get_centroid(y)**2 - 3 * get_centroid(y)
            ) / get_variance(y)**2

    calculation = dict(
        centroid=get_centroid,
        variance=get_variance,
        skew=get_skew,
        kurtosis=get_kurtosis
    )

    fft_abs = np.abs(np.fft.rfft(x))

    res = [calculation[config["aggtype"]](fft_abs) for config in param]
    return res
