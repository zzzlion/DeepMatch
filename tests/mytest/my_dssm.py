
import pandas as pd
from sklearn.utils import shuffle

# https://zhuanlan.zhihu.com/p/136253355


model_name = "DSSM"
samples_data = pd.read_csv("E:\github repo\DeepMatch\examples\samples_dssm.txt", sep="\t", header=None)
samples_data.columns = ["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id",
                        "label"]
# samples_data.head()
samples_data = shuffle(samples_data)
X = samples_data[["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id"]]
y = samples_data["label"]
# 转换数据存储格式
import numpy as np

train_model_input = {"user_id": np.array(X["user_id"]), \
                     "gender": np.array(X["gender"]), \
                     "age": np.array(X["age"]), \
                     "hist_movie_id": np.array([[int(i) for i in l.split(',')] for l in X["hist_movie_id"]]), \
                     "hist_len": np.array(X["hist_len"]), \
                     "movie_id": np.array(X["movie_id"]), \
                     "movie_type_id": np.array(X["movie_type_id"])}

train_label = np.array(y)

# 统计每个离散特征的词频量，构造特征参数
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepmatch.models import DSSM

embedding_dim = 32
SEQ_LEN = 50
user_feature_columns = [SparseFeat('user_id', max(samples_data["user_id"]) + 1, embedding_dim),
                        SparseFeat("gender", max(samples_data["gender"]) + 1, embedding_dim),
                        SparseFeat("age", max(samples_data["age"]) + 1, embedding_dim),
                        VarLenSparseFeat(
                            SparseFeat('hist_movie_id', max(samples_data["movie_id"]) + 1, embedding_dim,
                                       embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                        ]

item_feature_columns = [SparseFeat('movie_id', max(samples_data["movie_id"]) + 1, embedding_dim),
                        SparseFeat('movie_type_id', max(samples_data["movie_type_id"]) + 1, embedding_dim)]

model = DSSM(user_feature_columns, item_feature_columns)
model.compile(optimizer='adagrad', loss="binary_crossentropy", metrics=['accuracy'])

history = model.fit(train_model_input, train_label,
                    batch_size=256, epochs=10, verbose=1, validation_split=0.2)

# check_model(model, model_name, X, y)
