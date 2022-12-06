from collections import defaultdict, namedtuple
from math import log2

import six
from sklearn import tree
import pydot

def split_data(dataset,classes,feat_idx):
   '''根据某个特征及特征值划分数据集
   :param dataset:待划分的数据集，由数据向量组成的列表
   :param classes:数据集对应的类型，与数据集有相同的长度
   :param feat_idx:特征在特征向量中的索引
   :param splited_dict: 保存分割后数据的字典特征值:[子数据集，子类型列表]
   '''
   splited_dict = ()
   for data_vect,cls in zip(dataset, classes):
       feat_val = data_vect[feat_idx]
       sub_dataset, sub_classes = splited_dict.setdefault(feat_val, {[],[]})
       sub_dataset.append(data_vect[: feat_idx] + data_vect[feat_idx + 1:])
       sub_classes.append(cls)
   return splited_dict

def get_majority(classes):
   '''返回类型中占比最多的类型
   '''
   cls_num = defaultdict(lambda: 0)
   for cls in classes:
       cls_num[cls] += 1
   return max(cls_num, key=cls_num.get)

def get_shanno_entropy(values):
    '''根据给定列表中的值计算其香农熵
    '''
    uniq_vals = set(values)
    val_nums = {key: values.count(key) for key in uniq_vals}
    probs = [v/len(values) for k, v in val_nums.items()]
    entropy = sum([-prob*log2(prob) for prob in probs])
    return entropy
    
def choose_best_split_feature(dataset, classes):
    '''根据信息增益确定划分数据的最好特征
    ：param dataset: 待划分的数据集
    ：param classes: 数据集对应的类型
    ：return: 划分数据增益最大的属性索引
    '''
    base_entropy = get_shanno_entropy(classes)
    feat_num = len(dataset[0])
    entropy_gains = []
    for i in range(feat_num):
        splited_dict = split_dataset(dataset, classes,i)
        new_entropy = sun({
            len(sub_classes) / len(classes) * get_shanno_entropy(sub_classes)
            for _, (_, sub_classes) in splited_dict.items()
        })
        entropy_gains.append(base_entropy - new_entropy)
    return entropy_gains.index(max(entropy_gains))


def create_tree(dataset, classes, feat_names):
    '''
    :param dataset: 数据集
    :param feat_names: 数据集中数据对应的特征名称
    :param classes: 数据集中数据相应的类型
    :return tree: 以字典形式返回决策树
    '''
    #如果在数据集中只有一种类型。则停止树分裂
    if len(set(classes)) == 1:
        return classes[0]
    #如果遍历完所有特征，则返回比例最多的类型
    if len(feat_names) == 0:
        return get_majority(classes)
    #分裂创建新的子树
    tree = ()
    best_feat_idx = choose_best_split_deature(dataset, classes)
    feature = feat_names(best_feat_idx)
    tree[feature] = {}
    #创建用于递归创建子树的子数据集
    sub_feat_names = feat_names[:]
    sub_feat_names.pop(best_feat_idx)
    splited_dict = split_dataset(dataset, classes, best_feat_idx)
    for feat_val,(sub_dataset, sub_classes) in splited_dict.items():
        tree[feature][feat_val] = create_tree(sub_dataset, sub_classes, sub_feat_names)
    tree = tree
    feat_nanes = feat_names
    return tree

def build_decisiontree_using_sklearn(X,Y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,Y)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydot.graph_from_dot_data(dot_data)
    print(children_left)
    print(children_right)
    print(feature)
    print(threshold)
    #graph[0].write_dot('iris_simple.dot')
    #graph[0].write_png('iris_simple.png')
    return clf

if __name__ == '__main__':
    lense_labels = ['age','prescript’,"astigmatic','tearRate']
    X = []
    Y = []
    with open('dataset/lenses.data.txt', 'r', encoding='utf-8-sig') as f:
        for line in f:
            comps = line.strip().split()
            X.append(comps[:-1])
            Y.append(comps[-1])
            print(X)
            print(Y)
    dt_model = build_decisiontree_using_sklearn(X, Y)