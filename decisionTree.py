import sys
import os
import csv
import math


def train_and_test(train_input, test_input, depth, train_out, test_out, metrics):
    data = handle_data(train_input)
    train_label = data[0]
    train_feats = data[1]
    train_tags = data[2]

    tree = decisionTreeTrain(train_label, train_feats, train_tags,
                             0, int(depth))
    # printTree(tree,0)

    train_erro = test(tree, train_input, train_out)
    test_erro = test(tree, test_input, test_out)

    str = "error(train): {}\nerror(test): {}".format(train_erro, test_erro)
    f = open(metrics, 'w')
    f.write(str)

    f.close()


##
## Helper functions


# handle_data: input an .csv file, return its labels, features and tags
def handle_data(train_input):
    labels = []
    feats = []
    features = {}

    with open(train_input, 'r') as csvfile:
        reader = csv.reader(csvfile)
        print(reader)

        for row in reader:
            line = len(row)

            if (len(feats) == 0):
                for i in range(line - 1):
                    feats.append([row[i]])

            else:
                for i in range(line - 1):
                    feats[i].append(row[i])

                labels.append(row[-1])

        for i in range(line - 1):
            temp = feats[i].pop(0)
            features[temp] = feats[i]

    csvfile.close()

    tags = list(set(labels))

    return [labels, features, tags]


# decisionTreeTrain: implementation of ID3
# input labels, features, tags, max_depths, return decision tree
def decisionTreeTrain(labels, features, tags, cur_depth, max_depth):
    tag0_num, tag1_num = count_num(labels, tags)

    if (tag0_num > tag1_num):
        guess = tags[0]
        guess_num = tag0_num
    else:
        guess = tags[1]
        guess_num = tag1_num

    # base case: no need to split further
    if (guess_num == len(labels)):
        return Tree(guess)

    # base case: cannot split further
    elif (len(features) == 0):
        return Tree(guess)

    # do not split over max_depth
    elif (cur_depth > max_depth):
        return Tree(guess)

    else:
        n_labels = []
        y_labels = []
        score = -1
        feature = []
        for i in features:
            # the accuracy we would get if we only queried on i
            cur_score, cur_n_labels, cur_y_labels = info_gain(features[i], labels)

            if (cur_score >= score):
                score = cur_score
                n_labels = cur_n_labels
                y_labels = cur_y_labels
                feature = i

        cur_depth += 1
        n_features, y_features = split_features(feature, features)


        left = decisionTreeTrain(n_labels, n_features, tags, cur_depth, max_depth)
        right = decisionTreeTrain(y_labels, y_features, tags, cur_depth, max_depth)

        return Tree(tags[0], left, right, feature)


# test the result of the tree, return the error rate
def test(tree, test_input, output):
    feats = []
    data = []
    count = 0
    total = 0

    with open(test_input, 'r') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:

            line = len(row)

            if (total == 0):
                for i in range(line - 1):
                    feats.append(row[i])

            else:
                d = {}
                for i in range(line - 1):
                    d[feats[i]] = row[i]

                label = decisionTreeTest(tree, d)
                data.append(label + '\n')
                if (label != row[-1]):
                    count += 1

            total += 1

    str = "".join(data)
    f = open(output, 'w')
    f.write(str)

    csvfile.close()
    f.close()

    return float(count) / (total - 1)


# decisionTreeTest takes in the decision tree and a dict of features
# return the predicted label
def decisionTreeTest(tree, d):
    if (tree.isLeaf()):
        return tree.tag
    else:
        if (d[tree.feature] == 'n'):
            return decisionTreeTest(tree.left, d)
        else:
            return decisionTreeTest(tree.right, d)


# printTree: print out the decision tree
def printTree(tree, depth):
    if (tree.isLeaf()):
        print(' ' * depth, tree.tag)
    else:
        print(' ' * depth, tree.tag, ': ', tree.feature)
        printTree(tree.left, depth + 1)
        printTree(tree.right, depth + 1)


# calculate information gain for the specific feature
# Gain(T, X) = Entropy(T) - Entropy(T, X)
# Entropy(T, X) = Sum(c)(P(c)E(c))
def info_gain(feature, labels):
    n_labels = []
    y_labels = []
    tags = list(set(labels))

    for i in range(len(feature)):
        if (feature[i] == 'n'):
            n_labels.append(labels[i])
        else:
            y_labels.append(labels[i])

    p0 = float(len(n_labels)) / len(labels)
    p1 = float(len(y_labels)) / len(labels)

    return get_entropy(labels, tags) - (p0 * get_entropy(n_labels, tags) +
                                        p1 * get_entropy(y_labels, tags)), n_labels, y_labels


# get entropy of labels
#
def get_entropy(labels, tags):
    tag0_num, tag1_num = count_num(labels, tags)

    if (tag0_num == 0 or tag1_num == 0):
        return 0

    p0 = float(tag0_num) / len(labels)
    p1 = float(tag1_num) / len(labels)

    return -1 * (p0 * math.log(p0, 2) + p1 * math.log(p1, 2))


def split_features(feature, features):
    n_features = {}
    y_features = {}
    len_list = len(features[feature])
    for i in features:
        if (i == feature):
            continue
        n_features[i] = []
        y_features[i] = []
        for j in range(len_list):
            if (features[feature][j] == "n"):
                n_features[i].append(features[i][j])
            else:
                y_features[i].append(features[i][j])

    return n_features, y_features


def count_num(labels, tags):
    tag0_num = 0
    tag1_num = 0

    for i in labels:
        if (i == tags[0]):
            tag0_num += 1
        else:
            tag1_num += 1

    return tag0_num, tag1_num


# Tree Class
class Tree(object):
    def __init__(self, tag, left=None, right=None, feature=None):
        self.tag = tag
        self.left = left
        self.right = right
        self.feature = feature

    def isLeaf(self):
        if (self.left == None and self.right == None):
            return True
        else:
            return False


##
# Main function
if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    depth = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics = sys.argv[6]

    train_and_test(train_input, test_input, depth, train_out, test_out, metrics)
