import pprint
import numpy as np


def gini_impurity_one_side(labels_values, num_of_unique_labels):
    rounded_values = list(map(lambda value: 0 if value <= 0 else round(value), labels_values))
    sum_over_labels = sum(rounded_values)

    s = 0
    if sum_over_labels <= 0:
        return 0
    for l in range(num_of_unique_labels):
        s += (rounded_values[l] / sum_over_labels) ** 2

    impurity = (1 - s) * sum_over_labels
    return impurity


if __name__ == "__main__":
    num_of_unique_labels = 3
    num_of_features = 5
    num_of_thresholds = 40

    result = [[]] * 10

    labels_values_left = [[[0 for k in range(num_of_unique_labels)] for j in range(num_of_thresholds)] for i in range(num_of_features)]
    labels_values_right = [[[0 for k in range(num_of_unique_labels)] for j in range(num_of_thresholds)] for i in range(num_of_features)]

    ranges = [[0, 400], [0, 400], [0, 400], [0, 400], [0, 400], [0, 400], [0, 400], [0, 400], [-1, 0]]

# 2.220446E-16

    cnt = 0
    rng = np.random.RandomState(1)

    for r in ranges:
        min_impurity_result = None
        min_impurity_feature_index = None
        min_impurity_threshold_index = None

        for f in range(num_of_features):
            for t in range(num_of_thresholds):
                for l in range(num_of_unique_labels):
                    labels_values_left[f][t][l] = round(rng.randint(r[0] * 10000, r[1] * 10000)/10000, 4)
                    labels_values_right[f][t][l] = round(rng.randint(r[0] * 10000, r[1] * 10000)/10000, 4)

        for feature_index in range(num_of_features):
            for threshold_index in range(num_of_thresholds):
                gini_impurity_left = gini_impurity_one_side(labels_values_left[feature_index][threshold_index], num_of_unique_labels)
                gini_impurity_right = gini_impurity_one_side(labels_values_right[feature_index][threshold_index], num_of_unique_labels)
                gini_impurity = gini_impurity_left + gini_impurity_right

                if min_impurity_result is None or gini_impurity < min_impurity_result:
                    min_impurity_result = gini_impurity
                    min_impurity_feature_index = feature_index
                    min_impurity_threshold_index = threshold_index

        result[cnt] = [min_impurity_feature_index, min_impurity_threshold_index]
        cnt += 1

    min_impurity_result = None
    min_impurity_feature_index = None
    min_impurity_threshold_index = None

    for f in range(num_of_features):
        for t in range(num_of_thresholds):
            for l in range(num_of_unique_labels):
                labels_values_left[f][t][l] = 2.220446E-30
                labels_values_right[f][t][l] = 2.220446E-30

    for feature_index in range(num_of_features):
        for threshold_index in range(num_of_thresholds):
            gini_impurity_left = gini_impurity_one_side(labels_values_left[feature_index][threshold_index], num_of_unique_labels)
            gini_impurity_right = gini_impurity_one_side(labels_values_right[feature_index][threshold_index], num_of_unique_labels)
            gini_impurity = gini_impurity_left + gini_impurity_right

            if min_impurity_result is None or gini_impurity < min_impurity_result:
                min_impurity_result = gini_impurity
                min_impurity_feature_index = feature_index
                min_impurity_threshold_index = threshold_index

    result[cnt] = [min_impurity_feature_index, min_impurity_threshold_index]

    pprint.pprint(result)
