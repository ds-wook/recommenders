# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import logging
import numpy as np
import category_encoders as ce
from tqdm import tqdm
import collections
import gc

import lightgbm as lgb


def unpackbits(x, num_bits):
    """Convert a decimal value numpy.ndarray into multi-binary value numpy.ndarray ([1,2]->[[0,1],[1,0]])

    Args:
        x (numpy.ndarray): Decimal array.
        num_bits (int): The max length of the converted binary value.
    """
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = 2 ** np.arange(num_bits).reshape([1, num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])


class NumEncoder(object):
    """Encode all the categorical features into numerical ones by sequential label encoding, sequential count encoding,
    and binary encoding. Additionally, it also filters the low-frequency categories and fills the missing values.
    """

    def __init__(self, cate_cols, nume_cols, label_col, threshold=10, thresrate=0.99):
        """Constructor.

        Args:
            cate_cols (list): The columns of categorical features.
            nume_cols (list): The columns of numerical features.
            label_col (object): The column of Label.
            threshold (int): The categories whose frequency is lower than the threshold will be filtered (be treated
                as "<LESS>").
            thresrate (float): The (1.0 - thersrate, default 1%) lowest-frequency categories will also be filtered.
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFO] %(message)s")
        self.label_name = label_col
        self.cate_cols = cate_cols
        self.dtype_dict = {}
        for item in cate_cols:
            self.dtype_dict[item] = "str"
        for item in nume_cols:
            self.dtype_dict[item] = "float"
        self.nume_cols = nume_cols
        self.tgt_nume_cols = []
        self.encoder = ce.ordinal.OrdinalEncoder(cols=cate_cols)
        self.threshold = threshold
        self.thresrate = thresrate

        self.save_cate_avgs = {}
        self.save_value_filter = {}
        self.save_num_embs = {}
        self.Max_len = {}
        self.samples = 0

    def fit_transform(self, df):
        """Input a training set (pandas.DataFrame) and return the converted 2 numpy.ndarray (x,y).

        Args:
            df (pandas.DataFrame): Input dataframe

        Returns:
            numpy.ndarray, numpy.ndarray: New features and labels.
        """
        df = df.astype(dtype=self.dtype_dict)
        self.samples = df.shape[0]
        logging.info("Filtering and fillna features")
        for item in tqdm(self.cate_cols):
            value_counts = df[item].value_counts()
            num = value_counts.shape[0]
            self.save_value_filter[item] = list(
                value_counts[: int(num * self.thresrate)][
                    value_counts > self.threshold
                ].index
            )
            rm_values = set(value_counts.index) - set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: "<LESS>" if x in rm_values else x)
            df[item] = df[item].fillna("<UNK>")
            del value_counts
            gc.collect()

        for item in tqdm(self.nume_cols):
            df[item] = df[item].fillna(df[item].mean())
            self.save_num_embs[item] = {"sum": df[item].sum(), "cnt": df[item].shape[0]}

        logging.info("Ordinal encoding cate features")
        # ordinal_encoding
        df = self.encoder.fit_transform(df)

        logging.info("Target encoding cate features")
        # dynamic_targeting_encoding
        for item in tqdm(self.cate_cols):
            feats = df[item].values
            labels = df[self.label_name].values
            feat_encoding = {"mean": [], "count": []}
            self.save_cate_avgs[item] = collections.defaultdict(lambda: [0, 0])
            for idx in range(self.samples):
                cur_feat = feats[idx]
                if cur_feat in self.save_cate_avgs[item]:
                    feat_encoding["mean"].append(
                        self.save_cate_avgs[item][cur_feat][0]
                        / self.save_cate_avgs[item][cur_feat][1]
                    )
                    feat_encoding["count"].append(
                        self.save_cate_avgs[item][cur_feat][1] / idx
                    )
                else:
                    feat_encoding["mean"].append(0)
                    feat_encoding["count"].append(0)
                self.save_cate_avgs[item][cur_feat][0] += labels[idx]
                self.save_cate_avgs[item][cur_feat][1] += 1
            df[item + "_t_mean"] = feat_encoding["mean"]
            df[item + "_t_count"] = feat_encoding["count"]
            self.tgt_nume_cols.append(item + "_t_mean")
            self.tgt_nume_cols.append(item + "_t_count")

        logging.info("Start manual binary encoding")
        rows = None
        for item in tqdm(self.nume_cols + self.tgt_nume_cols):
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1, 1))
            else:
                rows = np.concatenate([rows, feats.reshape((-1, 1))], axis=1)
            del feats
            gc.collect()
        for item in tqdm(self.cate_cols):
            feats = df[item].values
            Max = df[item].max()
            bit_len = len(bin(Max)) - 2
            samples = self.samples
            self.Max_len[item] = bit_len
            res = unpackbits(feats, bit_len).reshape((samples, -1))
            rows = np.concatenate([rows, res], axis=1)
            del feats
            gc.collect()
        trn_y = np.array(df[self.label_name].values).reshape((-1, 1))
        del df
        gc.collect()
        trn_x = np.array(rows)
        return trn_x, trn_y

    # for test dataset
    def transform(self, df):
        """Input a testing / validation set (pandas.DataFrame) and return the converted 2 numpy.ndarray (x,y).

        Args:
            df (pandas.DataFrame): Input dataframe

        Returns:
            numpy.ndarray, numpy.ndarray: New features and labels.
        """
        df = df.astype(dtype=self.dtype_dict)
        samples = df.shape[0]
        logging.info("Filtering and fillna features")
        for item in tqdm(self.cate_cols):
            value_counts = df[item].value_counts()
            rm_values = set(value_counts.index) - set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: "<LESS>" if x in rm_values else x)
            df[item] = df[item].fillna("<UNK>")

        for item in tqdm(self.nume_cols):
            mean = self.save_num_embs[item]["sum"] / self.save_num_embs[item]["cnt"]
            df[item] = df[item].fillna(mean)

        logging.info("Ordinal encoding cate features")
        # ordinal_encoding
        df = self.encoder.transform(df)

        logging.info("Target encoding cate features")
        # dynamic_targeting_encoding
        for item in tqdm(self.cate_cols):
            avgs = self.save_cate_avgs[item]
            df[item + "_t_mean"] = df[item].map(
                lambda x: avgs[x][0] / avgs[x][1] if x in avgs else 0
            )
            df[item + "_t_count"] = df[item].map(
                lambda x: avgs[x][1] / self.samples if x in avgs else 0
            )

        logging.info("Start manual binary encoding")
        rows = None
        for item in tqdm(self.nume_cols + self.tgt_nume_cols):
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1, 1))
            else:
                rows = np.concatenate([rows, feats.reshape((-1, 1))], axis=1)
            del feats
            gc.collect()
        for item in tqdm(self.cate_cols):
            feats = df[item].values
            bit_len = self.Max_len[item]
            res = unpackbits(feats, bit_len).reshape((samples, -1))
            rows = np.concatenate([rows, res], axis=1)
            del feats
            gc.collect()
        vld_y = np.array(df[self.label_name].values).reshape((-1, 1))
        del df
        gc.collect()
        vld_x = np.array(rows)
        return vld_x, vld_y


class LightGBMRanker:
    """LightGBM-based ranker for recommendation tasks.

    Wraps LightGBM training, inference, save, and load into a single class
    that follows a sklearn-like interface. Supports both binary classification
    (CTR prediction) and learning-to-rank objectives (lambdarank, etc.).

    Example::

        ranker = LightGBMRanker(params={"objective": "binary", "metric": "auc"})
        ranker.fit(train_x, train_y, valid_x, valid_y)
        scores = ranker.predict(test_x)
        ranker.save("model.lgb")

        loaded = LightGBMRanker.load("model.lgb")
        scores = loaded.predict(test_x)
    """

    DEFAULT_PARAMS = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 64,
        "min_data": 20,
        "boost_from_average": True,
        "num_threads": 4,
        "feature_fraction": 0.8,
        "learning_rate": 0.15,
    }

    def __init__(self, params=None, num_boost_round=100, early_stopping_rounds=20):
        """Constructor.

        Args:
            params (dict): LightGBM parameters. Merged with DEFAULT_PARAMS; keys
                in ``params`` take precedence.
            num_boost_round (int): Maximum number of boosting iterations.
            early_stopping_rounds (int): Stop training if no improvement for this
                many rounds on the validation set. Ignored when no validation set
                is provided.
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, train_x, train_y, valid_x=None, valid_y=None, categorical_feature=None):
        """Train the ranker.

        Args:
            train_x (numpy.ndarray or pandas.DataFrame): Training features.
            train_y (numpy.ndarray): Training labels, shape ``(n,)`` or ``(n, 1)``.
            valid_x (numpy.ndarray or pandas.DataFrame, optional): Validation features.
            valid_y (numpy.ndarray, optional): Validation labels.
            categorical_feature (list, optional): Column names or indices of
                categorical features passed to LightGBM. Defaults to ``"auto"``.

        Returns:
            LightGBMRanker: self
        """
        cat_feat = categorical_feature or "auto"
        lgb_train = lgb.Dataset(
            train_x, train_y.reshape(-1), params=self.params, categorical_feature=cat_feat
        )

        valid_sets = [lgb_train]
        callbacks = []
        if valid_x is not None and valid_y is not None:
            lgb_valid = lgb.Dataset(
                valid_x,
                valid_y.reshape(-1),
                reference=lgb_train,
                categorical_feature=cat_feat,
            )
            valid_sets.append(lgb_valid)
            if self.early_stopping_rounds:
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

        self.model = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks or None,
        )
        return self

    def predict(self, x):
        """Return ranking scores for the given features.

        Args:
            x (numpy.ndarray or pandas.DataFrame): Input features.

        Returns:
            numpy.ndarray: Predicted scores, shape ``(n,)``.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() or load() first.")
        return self.model.predict(x)

    def save(self, path):
        """Save the trained model to a file.

        Args:
            path (str): Destination file path (e.g. ``"model.lgb"``).
        """
        if self.model is None:
            raise RuntimeError("No trained model to save.")
        self.model.save_model(path)
        logging.info("LightGBMRanker model saved to %s", path)

    @classmethod
    def load(cls, path, params=None, num_boost_round=100, early_stopping_rounds=20):
        """Load a previously saved model from a file.

        Args:
            path (str): Path to the saved model file.
            params (dict, optional): LightGBM parameters stored on the instance
                (not used for inference, but available for reference).
            num_boost_round (int): Stored on the instance for reference.
            early_stopping_rounds (int): Stored on the instance for reference.

        Returns:
            LightGBMRanker: Instance with the loaded model ready for prediction.
        """
        ranker = cls(
            params=params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        ranker.model = lgb.Booster(model_file=path)
        logging.info("LightGBMRanker model loaded from %s", path)
        return ranker
