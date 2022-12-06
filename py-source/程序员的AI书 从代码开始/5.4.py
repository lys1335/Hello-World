import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

movies_path = 'dataset/ml-latest-small/movies.csv'
ratings_path = 'dataset/ml-latest-small/ratings.csv'


# 将rating转换成0、1分类
def convert_rating_2_labels(ratings):
    label = []
    ratings_list = ratings.values.tolist()
    for rate in ratings_list:
        if rate >= 3.0:
            label.append(1)
        else:
            label.append(0)
    return label


# 将genres转换成One-hot特征
def convert_2_one_hot(df):
    genres_vals = df['genres'].values.tolist()
    genres_set = set()
    for row in genres_vals:
        genres_set.update(row.split('|'))
    genres_list = list(genres_set)
    row_num = 0
    df_new = pd.DataFrame(columns=genres_list)
    for row in genres_vals:
        init_genres_vals = [0] * len(genres_list)
        genres_names = row.split('|')
        for name in genres_names:
            init_genres_vals[genres_list.index(name)] = 1
        df_new.loc[row_num] = init_genres_vals
        row_num += 1
    df_update = pd.concat([df, df_new], axis=1)
    return df_update


# 构建逻辑回归模型
def training_lr(x, y):
    model = LogisticRegression(penalty='l2', C=1, solver='sag', max_iter=500, verbose=1, n_jobs=8)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    model.fit(x_train, y_train)
    train_pred = model.predict_proba(x_train)
    train_auc = roc_auc_score(y_train, train_pred[:, 1])

    test_pred = model.predict_proba(x_test)
    test_auc = roc_auc_score(y_test, test_pred[:, 1])

    # print(model.score())
    print('lr train auc score:' + str(train_auc))
    print('lr test auc score:' + str(test_auc))


# 读取数据
def load_data():
    movie_df = pd.read_csv(movies_path)
    rating_df = pd.read_csv(ratings_path)
    df_update = convert_2_one_hot(movie_df)
    df_final = pd.merge(rating_df, df_update, on='movieId')
    ratings = df_final['rating']
    df_final = df_final.drop(columns=['userId', 'movieId', 'timestamp', 'title', 'genres', 'rating'])
    labels = convert_rating_2_labels(ratings)
    trainx = df_final.values.tolist()
    return trainx, labels


if __name__ == '__main__':
    trainx, labels = load_data()
    training_lr(trainx, labels)
