#!/bin/python

import os
import gc

import pandas as pd
import numpy as np

from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.regularizers import l1
from keras import backend as K

from sklearn.preprocessing import power_transform, StandardScaler
from sklearn.decomposition import FactorAnalysis


class LSTMEncoderImputer:
    # LSTM autoencoder for time series data
    # Returns a vector representation of dataframe
    # Missing values are taken from the average of the LSTM
    # prediction vector for the 5 nearest neighbors

    def __init__(self, df, dict_nn_for_missing, value_limit=20, epochs=150, save_to_disk=False):
        self.df = df
        self.dict_nn_for_missing = dict_nn_for_missing

        # These many observations will be considered for training
        # Any observations beyond this will be truncated
        # Observations less than this will be 0 padded
        self.value_limit = value_limit

        # Miscellaneous options
        self.epochs = epochs
        self.save_to_disk = save_to_disk

        # Run through each feature
        # > Scale data
        # > Reshape into 3D array
        # > Feed into LSTM

        # Store feature arrays in this dict
        self.dict_item_arrays = {}

        # Prediction dataframe
        self.df_predict = pd.DataFrame()

    def regularize_array_size(self, array):
        if len(array) > self.value_limit:
            return array[:self.value_limit]

        elif len(array) < self.value_limit:
            t = self.value_limit - len(array)
            return np.pad(array, pad_width=(0, t), mode='constant')

        return array

    def scale_reshape_array(self, array):
        # Scaling and reshaping
        # Scaling must take place before reshaping
        # LSTMs take 3D input only - Samples / observations / features

        transformed_array = power_transform(array, method='yeo-johnson')
        reshaped_array = transformed_array.reshape(
            len(transformed_array), self.value_limit, 1)

        return reshaped_array

    def create_feature_vectors(self):
        # Get all values for each feature in the dataframe
        # divided by HADM_ID
        df_item_group = self.df.groupby('ITEMID')
        for item_group in df_item_group:

            # All time series arrays for this ITEMID
            item_arrays_per_hadm = []
            corresponding_hadms = []

            this_itemid = item_group[0]
            df_hadm_value = item_group[1]
            df_hadm_value = df_hadm_value.drop(
                'ITEMID', axis=1)
            df_hadm_value_g = df_hadm_value.groupby('HADM_ID')

            # Iterate through each HADM_ID in the df_hadm_value_g
            # object and create a vector
            for hadm_group in df_hadm_value_g:
                this_hadm = hadm_group[0]
                df_value = hadm_group[1]
                value_array = df_value.VALUE.values  # HAHK
                value_array = self.regularize_array_size(value_array)

                item_arrays_per_hadm.append(value_array)
                corresponding_hadms.append(this_hadm)

            self.dict_item_arrays[this_itemid] = (
                corresponding_hadms,
                np.array(item_arrays_per_hadm))

    def gaping_maw(self, hadm_ids, X):
        # Run once per feature
        # Create and feed the neural network
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(self.value_limit, 1)))
        model.add(RepeatVector(self.value_limit))
        model.add(LSTM(100, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer='adam', loss='mse')

        # Start training
        model.fit(X, X, epochs=self.epochs, validation_split=0.1)

        # Declare the LSTM input layer as the output as well
        model = Model(inputs=model.inputs, outputs=model.layers[0].output)

        # Make predictions for values in the original df
        prediction_dict = {}
        for this_hadm, this_array in zip(hadm_ids, X):
            prediction_dict[this_hadm] = model.predict(
                this_array.reshape(1, self.value_limit, 1))

        K.clear_session()  # Clear memory after each trained feature
        return prediction_dict

    def complete_predictions_for_feature(self, prediction_dict):
        # Fill up all values before averaging nearest neighbors
        # Once predictions are done, iterate over the missing HADMs
        missing_hadms_for_this_itemid = (
            set(self.df.HADM_ID.to_list()) - set(prediction_dict.keys()))

        for this_hadm in missing_hadms_for_this_itemid:

            valid_neighbors = 0
            array_sum = np.array([])
            nearest_neighbors = self.dict_nn_for_missing[this_hadm]

            for this_neighbor in nearest_neighbors:
                try:
                    prediction = prediction_dict[this_neighbor]
                    if array_sum.size == 0:
                        array_sum = prediction
                    else:
                        array_sum += prediction

                    valid_neighbors += 1
                except KeyError:  # In case the nearest neighbor has no prediction
                    continue

            average_prediction = array_sum / valid_neighbors
            prediction_dict[this_hadm] = average_prediction

        return prediction_dict  # This dictionary has ALL the predictions now

    def process_item_ids(self):
        for count, this_itemid in enumerate(self.dict_item_arrays):
            print('Feature:', count)

            hadm_ids, X = self.dict_item_arrays[this_itemid]  # HADMs + corresponding arrays
            X = self.scale_reshape_array(X)

            prediction_dict = self.gaping_maw(hadm_ids, X)
            prediction_dict = self.complete_predictions_for_feature(prediction_dict)

            # Create a dataframe of this prediction dictionary
            # and add it to self.df_predict
            columns = ['HADM_ID', this_itemid]
            df_predict_feature = pd.DataFrame(
                prediction_dict.items(), columns=columns).set_index('HADM_ID')

            # Save each dataframe to disk to save memory
            # Or process as usual
            if self.save_to_disk:
                os.makedirs('LSTMOut', exist_ok=True)
                filename = 'LSTMOut' + os.path.sep + 'LSTM_' + str(this_itemid)
                df_predict_feature.to_pickle(filename)
                gc.collect()
            else:
                if self.df_predict.empty:
                    self.df_predict = df_predict_feature
                else:
                    self.df_predict = self.df_predict.join(
                        df_predict_feature, how='inner')


# Dataframes need to have their indices set to the HADM_ID
class Encoder:
    def __init__(
            self, df_discrete, df_arrays,
            output_dimensions, epochs,
            discard=True,
            factor_analysis=True,
            scale=False):

        # Arrayed dfs will be flattened and joined to the discrete ones
        self.df_discrete = df_discrete
        self.df_arrays = df_arrays
        self.final_hadms = None

        # Delete columns of all 0s
        # Delete inf and na
        self.discard = discard

        # Perform maximal component factor analysis before starting
        # This has really fixed the exploding gradient problem for me
        self.factor_analysis = factor_analysis

        # Scale the input before processing
        self.scale = scale

        self.output_dimensions = output_dimensions
        self.epochs = epochs

        self.df_predict = None  # Middle layer
        self.df_predict1 = None  # Middle -1 layer
        self.train = None

        # Declare optimizer
        # Use for exploding gradients
        self._adam = Adam(clipnorm=1., learning_rate=0.000001)

    def process_data(self):
        # Flattened series from array df
        self.df_arrays = self.df_arrays.applymap(lambda x: x.flatten())
        s_flat = self.df_arrays.apply(lambda x: np.concatenate(x.values), axis=1)
        s_flat.name = 'ARRAYS'

        # Join this to the discrete df
        df = self.df_discrete.join(s_flat, how='inner')
        df = df.drop('ARRAYS', axis=1)

        self.final_hadms = df.index

        training_dimensionality = np.append(
            df.iloc[0].values.tolist(), s_flat.iloc[0]).shape[0]

        self.train = np.zeros((df.shape[0], training_dimensionality))
        for i in range(df.shape[0]):
            self.train[i] = np.append(
                df.iloc[i].to_list(),
                s_flat.iloc[i])

        if self.discard:
            print('Dropping all features with no values / invalid values')
            df = pd.DataFrame(self.train)
            df = df.loc[:, (df != 0).any(axis=0)]
            df = df.replace(np.inf, np.nan).dropna()
            self.train = df.values

        if self.factor_analysis:
            print('Peforming factor analysis @ maximal variability')
            df = pd.DataFrame(df.values)
            df = FactorAnalysis(df.shape[0]).fit_transform(df)
            self.train = df

        if self.scale:
            print('Scaling')
            self.train = power_transform(self.train, method='yeo-johnson')
            # self.train = StandardScaler().fit_transform(self.train)

    def gaping_maw(self):
        model = Sequential()
        dimensions = self.train.shape[1]
        model.add(Dense(
            self.output_dimensions * 8,
            input_dim=dimensions,
            activation='tanh'))
        model.add(Dense(self.output_dimensions * 4, activation='tanh'))
        model.add(Dense(self.output_dimensions * 2, activation='tanh'))
        model.add(Dense(self.output_dimensions, activation='tanh', activity_regularizer=l1(10e-5)))
        # model.add(Dense(self.output_dimensions, activation='tanh'))
        model.add(Dense(self.output_dimensions * 2, activation='tanh'))
        model.add(Dense(self.output_dimensions * 4, activation='tanh'))
        model.add(Dense(self.output_dimensions * 8, activation='tanh'))
        model.add(Dense(dimensions, activation='sigmoid'))

        model.compile(optimizer='adam', loss='mse')

        # Start training
        model.fit(
            self.train, self.train,
            epochs=self.epochs,
            batch_size=15,
            validation_split=0.15,
            shuffle=True)

        # Declare the LSTM input layer as the output as well
        model = Model(inputs=model.inputs, outputs=model.layers[3].output)
        model_slightly_bigger = Model(inputs=model.inputs, outputs=model.layers[2].output)

        # Generate predictions
        # Prediction creation required a 2D array
        predictions = model.predict(self.train)
        self.df_predict = pd.DataFrame(
            [self.final_hadms, predictions]).T
        self.df_predict.columns = ['HADM_ID', 'PREDICTIONS']
        self.df_predict = self.df_predict.set_index('HADM_ID')

        predictions = model_slightly_bigger.predict(self.train)
        self.df_predict1 = pd.DataFrame(
            [self.final_hadms, predictions]).T
        self.df_predict1.columns = ['HADM_ID', 'PREDICTIONS']
        self.df_predict1 = self.df_predict1.set_index('HADM_ID')

        K.clear_session()
