from abc import ABC

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from scipy.signal import savgol_filter

import pandas as pd
import numpy as np
from scipy.stats import gmean
from modAL.models import ActiveLearner

#  Importing modules from TensorFlow and Keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

from matplotlib.pyplot import plot

#  Save model
import h5py
import io
import copy

from validation import Validation
from pathlib import Path


from utilities import str2bool, rm_tree, perf_columns


class Model(ABC):
    def __init__(self, X_train, Y_train,
                 X_test, Y_test, X_validation,
                 Y_validation):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_validation = X_validation
        self.Y_validation = Y_validation
        self.test_performance = None
        self.validation_performance = None

    def run(self):
        self.train()
        self.validation()

    def train(self):
        pass

    def validation(self):
        pass

class DeepSCAMsModel(Model):

    def __init__(self, X_train, Y_train,
                 X_test, Y_test, X_validation,
                 Y_validation):
        super().__init__(X_train, Y_train,
                 X_test, Y_test, X_validation,
                 Y_validation)
        self.scaler2 = MinMaxScaler()
        self.Deep_SCAMS_model = None
        super().run()

    @staticmethod
    def format_for_DeepSCAMs(fitted_scaler, X):
        X_fitted = fitted_scaler.transform(X)
        return X_fitted

    def train(self):
        seed = 1234
        MLP = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
                            beta_2=0.999, early_stopping=False, epsilon=1e-08,
                            hidden_layer_sizes=(100, 1000, 1000), learning_rate='constant',
                            learning_rate_init=0.001, max_iter=200, momentum=0.9,
                            n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                            random_state=1234, shuffle=True, solver='sgd', tol=0.0001,
                            validation_fraction=0.1, verbose=False, warm_start=False)

        self.scaler2 = self.scaler2.fit(self.X_train)
        X_train = self.format_for_DeepSCAMs(self.scaler2, self.X_train)
        self.Deep_SCAMS_model = MLP.fit(X_train, self.Y_train.T)

    def validation(self):
        X_test = self.format_for_DeepSCAMs(self.scaler2, self.X_test)
        test_performance = Validation(self.Deep_SCAMS_model, X_test, self.Y_test, 'Test').results
        self.test_performance = test_performance
        X_validation = self.format_for_DeepSCAMs(self.scaler2, self.X_validation)
        validation_performance = Validation(self.Deep_SCAMS_model, X_validation,
                                            self.Y_validation, 'Validation').results
        self.validation_performance = validation_performance



def create_keras_model(shape=2256,
                       dropout=0.4):
    model = Sequential()
    model.add(Dense(500, input_dim=shape,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=["AUC"])
    return model


class KerasClassifier(tf.keras.wrappers.scikit_learn.KerasClassifier):
    """
    TensorFlow Keras API neural network classifier.

    Workaround the tf.keras.wrappers.scikit_learn.KerasClassifier serialization
    issue using BytesIO and HDF5 in order to enable pickle dumps.

    Adapted from: https://github.com/keras-team/keras/issues/4274#issuecomment-519226139
    """

    def __getstate__(self):
        state = self.__dict__
        if "model" in state:
            model = state["model"]
            model_hdf5_bio = io.BytesIO()
            with h5py.File(model_hdf5_bio, mode="w") as file:
                model.save(file)
            state["model"] = model_hdf5_bio
            state_copy = copy.deepcopy(state)
            state["model"] = model
            return state_copy
        else:
            return state

    def __setstate__(self, state):
        if "model" in state:
            model_hdf5_bio = state["model"]
            with h5py.File(model_hdf5_bio, mode="r") as file:
                state["model"] = tf.keras.models.load_model(file)
        self.__dict__ = state


class TensorFlowModel(Model):

    def __init__(self, X_train, Y_train,
                 X_test, Y_test, X_validation,
                 Y_validation):
        super().__init__(X_train, Y_train,
                         X_test, Y_test, X_validation,
                         Y_validation)
        self.TFModel = KerasClassifier(create_keras_model)
        self.scaler = RobustScaler(quantile_range=(25, 75))
        super().run()

    @staticmethod
    def format_for_model(fitted_scaler, X):
        X_fitted = fitted_scaler.transform(X)
        return X_fitted

    def train(self):
        self.scaler = self.scaler.fit(self.X_train)
        X_train = self.format_for_model(self.scaler, self.X_train)
        self.TFModel.fit(X_train, self.Y_train.T, epochs=50)

    def validation(self):
        X_test = self.format_for_model(self.scaler, self.X_test)
        test_performance = Validation(self.TFModel, X_test,
                                      self.Y_test, 'Test').results
        self.test_performance = test_performance

        X_validation = self.format_for_model(self.scaler, self.X_validation)
        validation_performance = Validation(self.TFModel,
                                            X_validation, self.Y_validation, 'Validation').results
        self.validation_performance = validation_performance


class ALModel(Model):
    def __init__(self, X_train, Y_train,
                 X_test, Y_test, X_validation,
                 Y_validation, n_queries,
                 results_dir,
                 q_strategy, iteration,
                 n_initial=50):
        super().__init__(X_train, Y_train,
                 X_test, Y_test, X_validation,
                 Y_validation)
        self.n_initial = n_initial
        self.n_queries = n_queries
        self.model = KerasClassifier(create_keras_model)
        self.scaler = RobustScaler(quantile_range=(25, 75))
        self.q_strategy = q_strategy
        self.iteration = iteration
        self.pipe = PipelineSW([('scaler', self.scaler),
                                ('tf_mlp', self.model)])
        self.initialize_al()
        self.final_model = None
        self.results_dir = results_dir
        self.run()

    @staticmethod
    def random_choice(X_train, n_initial):
        initial_idx = np.random.choice(range(len(X_train)),
                                       size=n_initial, replace=False)
        return initial_idx

    @staticmethod
    def gmen_no_nan(list_with_stats):
        g_v = gmean(list_with_stats)
        if not np.isnan(g_v):
            return g_v
        else:
            return 0

    def initialize_al(self):

        initial_idx = self.random_choice(self.X_train, self.n_initial)

        while len(set(self.Y_train[initial_idx])) != 2:  # Check if both classes are presented
            initial_idx = self.random_choice(self.X_train, self.n_initial)

        X, Y = self.X_train[initial_idx], self.Y_train[initial_idx]
        X_initial, y_initial = self.X_train[initial_idx], self.Y_train[initial_idx]
        X_pool, y_pool = np.delete(self.X_train, initial_idx, axis=0), \
                         np.delete(self.Y_train, initial_idx, axis=0)

        learner = ActiveLearner(
            estimator=self.pipe,
            query_strategy=self.q_strategy,
            X_training=X_initial, y_training=y_initial
        )


        # Calculate initial performance metrics on the test set
        performance_test = Validation(learner, self.X_test,
                                      self.Y_test, 'Test').results
        performance_test_l = [performance_test.iloc[0].tolist()]
        gmean_test = [self.gmen_no_nan(performance_test_l[0])]

        # Calculate initial performance metrics on the validation set
        performance_validation = Validation(learner, self.X_validation,
                                            self.Y_validation, 'Validation').results
        performance_validation_l = [performance_validation.iloc[0].tolist()]
        gmean_validation = [self.gmen_no_nan(performance_validation_l[0])]

        initial_data = [X, Y]
        pool = [X_pool, y_pool]

        return initial_data, pool, performance_test_l, performance_validation_l, \
               learner, [gmean_test, gmean_validation]

    def run(self):

        initial_data, pool, performance_test_l,\
        performance_validation_l, learner, gmeans = self.initialize_al()

        X, Y = initial_data
        X_pool, y_pool = pool
        gmean_test, gmean_validation = gmeans

        for i in range(self.n_queries - 1):
            query_idx, query_inst = learner.query(X_pool)
            ### Edit ###
            learner.teach(X_pool[query_idx], y_pool[query_idx], epochs=50)
            X = np.append(X, X_pool[query_idx], axis=0)
            Y = np.append(Y, y_pool[query_idx])
            X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), \
                             np.delete(y_pool, query_idx, axis=0)

            performance_t_iter = Validation(learner, self.X_test,
                                            self.Y_test, 'Test').results
            performance_t_iter_l = performance_t_iter.iloc[0].tolist()
            gmean_test.append(self.gmen_no_nan(performance_t_iter_l))
            performance_test_l.append(performance_t_iter_l)

            performance_v_iter = Validation(learner, self.X_validation,
                                            self.Y_validation, 'Validation').results
            performance_v_iter_l = performance_v_iter.iloc[0].tolist()
            ### Edit ###
            #learner.estimator[1].model.save(Path('/home/khali/scams/Results/N_SP1_TTS') / 'TF_MLP_AL_{}.h5'.format(i+1))
            gmean_validation.append(self.gmen_no_nan(performance_v_iter_l))
            performance_validation_l.append(performance_v_iter_l)
        performance_test_l = np.array(performance_test_l)
        performance_test_l[np.isnan(performance_test_l)] = 0
        np_test_per_init = pd.DataFrame(performance_test_l, columns=perf_columns)
        np_test_per_init.to_csv(self.results_dir / 'test_initial_stats.csv')

        # performance_test_l_m = savgol_filter(performance_test_l.T, 7, 3)
        # performance_test_l_m = performance_test_l_m.T
        # np_test_per_smoothed = pd.DataFrame(performance_test_l_m, columns=perf_columns)
        # np_test_per_smoothed.to_csv(self.results_dir / 'test_smoothed_stats.csv')

        self.test_performance = list(np.max(np.array(performance_test_l), axis=0))
        self.test_performance = pd.DataFrame([self.test_performance], index=["test performance"],
                                              columns=perf_columns)

        performance_validation_l = np.array(performance_validation_l)
        performance_validation_l[np.isnan(performance_validation_l)] = 0
        np_val_per_init = pd.DataFrame(performance_validation_l, columns=perf_columns)
        np_val_per_init.to_csv(self.results_dir / 'validation_initial_stats.csv')

        # performance_validation_l = savgol_filter(performance_validation_l.T, 7, 2)
        # performance_validation_l = performance_validation_l.T
        # np_val_per_smoothed = pd.DataFrame(performance_validation_l, columns=perf_columns)
        # np_val_per_smoothed.to_csv(self.results_dir / 'validation_smoothed_stats.csv')

        self.validation_performance = list(np.max(np.array(performance_validation_l), axis=0))
        self.validation_performance = pd.DataFrame([self.validation_performance], index=["validation performance"],
                                              columns=perf_columns)
        integral_performace = np.concatenate((performance_test_l, performance_validation_l), axis=1)
        index_max_pef = np.argmax(gmean(integral_performace, axis=1))
        final_X, final_Y = X[0: index_max_pef + self.n_initial, ], Y[0: index_max_pef + self.n_initial, ]
        self.final_model = TensorFlowModel(final_X, final_Y, self.X_test,
                                           self.Y_test, self.X_validation, self.Y_validation).TFModel

        print('Argmax validation', np.argmax(np.array(gmean_validation)))
        print('Argmax test', np.argmax(np.array(gmean_test)))

class PipelineSW(Pipeline):
    def fit(self, X, y, sample_weight=None):
        """Fit and pass sample weights only to the last step"""
        if sample_weight is not None:
            kwargs = {self.steps[-1][0] + '__sample_weight': sample_weight}
        else:
            kwargs = {}
        return super().fit(X, y, **kwargs)