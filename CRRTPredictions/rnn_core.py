#!/bin/python

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense


class RNNCore:
    def __init__(self, df_input, df_outcome, list_hadms, start_intervals, stateful):
        # Dataframes
        # It is assumed that the input dataframe has an
        # equal number of observations for each patient,
        # that non-numeric data has been dropped,
        # and that the data is time sorted
        self.df_input = df_input.reset_index()
        self.array_input = None
        self.df_training = None
        self.df_predictions = None

        # Outcome data
        usable_hadms = self.df_input['HADM_ID'].unique()
        self.df_outcome = df_outcome.query('HADM_ID in @usable_hadms')

        # Create 2 LSTM RNNs with the same architecture
        # Train one with the data, and copy its weights over to the other one
        # Use the 2nd one for making predictions - 8 of them
        self.training_model = None
        self.prediction_model = None

        # Train over all patients or a list of patients
        self.list_hadms = list_hadms

        # Use data upto this interval for training
        self.start_intervals = start_intervals

        # Retain state
        self.stateful = stateful

        # Preprocessing
        self.shape = None
        self.scaler = None
        self.decomposition = None  # TODO

        # Store final predictions here
        self.dict_predictions = {}

    def select_data(self, outcome_variable):
        # TODO - See what results are like for 0 padded data in case
        # no restriction is performed
        # Data restriction
        if self.start_intervals:
            self.df_input = self.df_input.groupby('HADM_ID').head(self.start_intervals)

        # Patient selection
        # Go with the provided list of patients, OR
        # Select 2 patients at random - one with the outcome, one without
        if self.list_hadms:
            self.df_training = self.df_input.query('HADM_ID in @self.list_hadms')
            self.df_predictions = self.df_training.copy()
        else:
            patient_outcome_yes = self.df_outcome[
                self.df_outcome[outcome_variable] == True].sample()['HADM_ID']
            patient_outcome_no = self.df_outcome[
                self.df_outcome[outcome_variable] == False].sample()['HADM_ID']

            patients = (patient_outcome_yes, patient_outcome_no)
            self.df_training = self.df_input.query('HADM_ID not in @patients')
            self.df_predictions = self.df_input.query('HADM_ID in @patients')

        # Create a variable that holds the shape of the training data
        patients = self.df_training['HADM_ID'].unique().shape[0]
        timesteps = int(self.df_training.shape[0] / patients)
        features = self.df_training.shape[1] - 1
        self.shape = (patients, timesteps, features)

    def preprocess(self, df, shape=None):
        # HADM_ID is no longer needed
        df = df.drop('HADM_ID', axis=1)

        # Scaling
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(df)

        # Reshaping must take place on an array
        # It goes per the usual (Sequences, Timesteps, Features)
        # Sequences = Number of patients
        # This data is stored in self.shape

        new_shape = (self.shape[0], self.shape[1] - 1, self.shape[2])
        if shape:
            new_shape = shape

        try:
            reshaped_array = np.reshape(scaled, new_shape)
        except (TypeError, ValueError) as e:
            # Will error out if not a whole number or the reshape
            # doesn't feel like working
            breakpoint()
            print('Boss, the reshapes!', e.args)
            exit(1)

        return reshaped_array

    def create_training_data(self):
        # Create X and y datasets by shifting
        timesteps = self.shape[1]

        X = self.df_training.groupby('HADM_ID').head(timesteps - 1)
        y = self.df_training.groupby('HADM_ID').tail(timesteps - 1)

        X = self.preprocess(X)
        y = self.preprocess(y)

        return X, y

    def gaping_maw(self, X, y, epochs):
        input_shape = (self.shape[1] - 1, self.shape[2])

        self.training_model = Sequential()
        self.training_model.add(
            LSTM(100, input_shape=input_shape, return_sequences=True))
        self.training_model.add(
            LSTM(100, return_sequences=True))
        self.training_model.add(
            Dense(self.shape[2]))

        self.training_model.compile(optimizer='adam', loss='mse')
        self.training_model.fit(
            X, y, epochs=epochs, batch_size=64, shuffle=False, validation_split=0.1)

    def retching_maw(self):
        batch_input_shape = (1, None, self.shape[2])

        self.prediction_model = Sequential()
        self.prediction_model.add(
            LSTM(100, return_sequences=True, stateful=True, batch_input_shape=batch_input_shape))
        self.prediction_model.add(
            LSTM(100, return_sequences=False, stateful=True))
        self.prediction_model.add(
            Dense(self.shape[2]))

        self.prediction_model.set_weights(self.training_model.get_weights())

    def predict(self, n_predictions):
        df_g = self.df_predictions.groupby('HADM_ID')
        for this_group in df_g:

            this_hadm = this_group[0]
            this_df = this_group[1]

            pred_shape = (1, self.shape[1], self.shape[2])
            X = self.preprocess(this_df, pred_shape)

            all_predictions = []
            first_prediction = self.prediction_model.predict(X).reshape(1, 1, self.shape[2])
            all_predictions.append(first_prediction)

            for _ in range(n_predictions - 1):  # First prediction has already been made
                this_prediction = self.prediction_model.predict(
                    all_predictions[-1]).reshape(1, 1, self.shape[2])
                all_predictions.append(this_prediction)

            # Store predictions
            self.dict_predictions[this_hadm] = all_predictions

            # Reset state after each batch of predictions
            self.prediction_model.reset_states()

    def collate_predictions(self, scale=False):
        # Create dataframes of predictions
        # After an inverse transform
        dict_pred_final = {}

        for this_hadm in self.dict_predictions:
            predictions = self.dict_predictions[this_hadm]

            df_pred = pd.DataFrame()
            for this_prediction in predictions:
                df_this_pred = pd.DataFrame(
                    np.reshape(this_prediction, (1, self.shape[2])))
                df_pred = df_pred.append(df_this_pred)

            if scale:
                arr_pred = self.scaler.inverse_transform(df_pred)
                df_pred_final = pd.DataFrame(
                    arr_pred, columns=self.df_training.columns.drop('HADM_ID'))
            else:
                df_pred_final = df_pred.copy()
                df_pred_final.columns = self.df_training.columns.drop('HADM_ID')

            dict_pred_final[this_hadm] = df_pred_final

        return dict_pred_final
