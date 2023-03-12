#include "../src/decision_tree_node.h"
#include "../src/tree_eval_server.h"
#include "../src/soft_if.h"
#include "../src/fhe_client.h"
#include "../src/tree_train_server.h"
#include "../src/utils.h"
#include <fstream>
#include <utility>
#include <gtest/gtest.h>

/**
 *
 * @b Description:
 * Class that contains implementations over plaintext of tree training and evaluation, to be used as reference for testing
 * the same algorithms over encrypted data. The tests assume this class methods as 'source of truth'
 *
 */
class PlainMethods {
public:
    static void generate_gini_test_vector() {
        int labels_count = 3;
        int features_count = 5;
        int thetas_count = 40;
        mt19937 gen(1);
        vector<vector<int>> ranges{{0, 400}, {0, 400}, {0, 400}, {0, 400}, {0, 400}, {0, 400}, {0, 400}, {0, 400}, {-1, 0}};
        size_t n_tests = 10;

        // Generate data
        vector_3d<double> node_sums_left(features_count,vector_2d<double>(thetas_count,vector<double>(labels_count)));
        vector_3d<double> node_sums_right(features_count,vector_2d<double>(thetas_count,vector<double>(labels_count)));
        vector<pair<vector_3d<double>, vector_3d<double>>> node_sums_arr(n_tests);

        for (size_t i = 0; i < n_tests; i++) {
            for (size_t k = 0; k < features_count; k++) {
                for (size_t s = 0; s < thetas_count; s++) {
                    for (size_t l = 0; l < labels_count; l++) {
                        if (i == n_tests - 1) {
                            node_sums_left[k][s][l] = 2.220446E-30;
                            node_sums_right[k][s][l] = 2.220446E-30;
                        }
                        else {
                            uniform_int_distribution<> dis(ranges[i][0] * 10000, ranges[i][1] * 10000);
                            node_sums_left[k][s][l] = (double)dis(gen) / 10000;
                            node_sums_right[k][s][l] = (double)dis(gen) / 10000;
                        }
                    }
                }
            }

            pair<vector_3d<double>, vector_3d<double>> tup = {node_sums_left, node_sums_right};
            node_sums_arr[i] = tup;
        }


        // Write to gini_test_vector.txt
        ofstream input_file;
        input_file.open("../tests/gini_test_utils/gini_test_vector.txt", ofstream::out | ofstream::trunc);

        for (size_t i = 0; i < n_tests; i++) {
            for (size_t k = 0; k < features_count; k++) {
                for (size_t s = 0; s < thetas_count; s++) {
                    for (size_t l = 0; l < labels_count; l++) {
                        input_file << node_sums_arr[i].first[k][s][l] << '\n';
                        input_file << node_sums_arr[i].second[k][s][l] << '\n';
                    }
                }
            }
        }

        input_file.close();
    }

    static vector<pair<vector_3d<double>, vector_3d<double>>> read_gini_test_vector() {
        int labels_count = 3;
        int features_count = 5;
        int thetas_count = 40;
        vector<vector<int>> ranges{{0, 400}, {0, 400}, {0, 400}, {0, 400}, {0, 400}, {0, 400}, {0, 400}, {0, 400}, {-1, 0}};
        string line;
        vector<pair<vector_3d<double>, vector_3d<double>>> node_sums_arr(10);
        ifstream output_file ("../tests/gini_test_utils/gini_test_vector.txt");

        for (size_t i = 0; i < 10; i++) {
            vector_3d<double> temp_left(features_count,vector_2d<double>(thetas_count,vector<double>(labels_count)));
            vector_3d<double> temp_right(features_count,vector_2d<double>(thetas_count,vector<double>(labels_count)));

            for (size_t k = 0; k < features_count; k++) {
                for (size_t s = 0; s < thetas_count; s++) {
                    for (size_t l = 0; l < labels_count; l++) {
                        getline(output_file,line);
                        temp_left[k][s][l] = stod(line);
                        getline(output_file,line);
                        temp_right[k][s][l] = stod(line);
                    }
                }
            }

            pair<vector_3d<double>, vector_3d<double>> tup = {temp_left, temp_right};
            node_sums_arr[i] = tup;
        }
        output_file.close();

        return node_sums_arr;
    }

    static double plain_soft_if(double input_x, const vector<double>& coeffs, const double free_coeff) {
        double true_result = 0.0;

        for (size_t j = 0; j < coeffs.size(); j++) {
            true_result += coeffs[j] * pow(input_x, j + 1);
        }

        true_result += free_coeff;

        return true_result;
    }

    static double plain_soft_if(double input_x, int input_poly_degree) {
        double free_coeff;
        vector<double> coeffs;
        int poly_degree = input_poly_degree;

        if (poly_degree > 0 && (poly_degree&(poly_degree-1)) == poly_degree)
        {
            cout << "Invalid poly degree for soft_if... initialized to 8..." << endl;
            poly_degree = 8;
        }
        else {
            poly_degree = poly_degree;
        }

        if (poly_degree == 8) {
            coeffs = COEFFS8;
            free_coeff = FREE_COEFF8;
        }
        else if (poly_degree == 16) {
            coeffs = COEFFS16;
            free_coeff = FREE_COEFF16;
        }
        else if (poly_degree == 32) {
            coeffs = COEFFS32;
            free_coeff = FREE_COEFF32;
        }
        else {
            coeffs = COEFFS8;
            free_coeff = FREE_COEFF8;
        }

        return plain_soft_if(input_x, coeffs, free_coeff);
    }

    static void build_base_tree(vector<shared_ptr<decision_tree_node>>& tree, vector<Sample<double>>& data, vector<double>& weights, vector<double>& splits, size_t height, int soft_if_poly_degree, size_t node_count = 0) {
        if (height == 0) {
            vector<double> label_scores(access_labels(data[0]).size(), 0.0);
            for (size_t i = 0; i < data.size(); i++) {
                for (size_t l = 0; l < access_labels(data[i]).size(); l++) {
                    label_scores[l] += weights[i] * access_labels(data[i])[l];
                }
            }

            tree[node_count] = make_shared<decision_tree_node>(node_count, label_scores);
        }
        else {
            vector<vector<vector<double>>> left_sums(access_features(data[0]).size(),
                                                     vector<vector<double>>(splits.size(),
                                                                            vector<double>(
                                                                                    access_labels(
                                                                                            data[0]).size(), 0.0)));

            vector<vector<vector<double>>> right_sums(access_features(data[0]).size(),
                                                      vector<vector<double>>(splits.size(),
                                                                             vector<double>(
                                                                                     access_labels(
                                                                                             data[0]).size(), 0.0)));

            vector<vector<vector<double>>> soft_ifs(data.size(),
                                                    vector<vector<double>>(access_features(data[0]).size(),
                                                                           vector<double>(splits.size(), 0.0)));

            //Step 1
            for (size_t i = 0; i < data.size(); i++) {
                for (size_t k = 0; k < access_features(data[i]).size(); k++) {
                    for (size_t s = 0; s < splits.size(); s++) {

                        double x_minus_theta = access_features(data[i])[k] - splits[s];
                        double soft_if_result = plain_soft_if(x_minus_theta, soft_if_poly_degree);

                        //STRONG IF:
//                    soft_if_result = x_minus_theta > 0.0 ? 1.0 : 0.0;

                        soft_ifs[i][k][s] = soft_if_result;
                        double weighted_soft_if = weights[i] * soft_if_result;
                        double weighted_inv_soft_if = weights[i] * (1.0 - soft_if_result);

                        for (size_t l = 0; l < access_labels(data[i]).size(); l++) {
                            right_sums[k][s][l] += weighted_soft_if * access_labels(data[i])[l];
                            left_sums[k][s][l] += weighted_inv_soft_if * access_labels(data[i])[l];
                        }
                    }
                }
            }

            //Step 2 (Oracle)
            for (size_t k = 0; k < access_features(data[0]).size(); k++) {
                for (size_t s = 0; s < splits.size(); s++) {
                    for (size_t l = 0; l < access_labels(data[0]).size(); l++) {
                        if (left_sums[k][s][l] < 0.0) {
                            left_sums[k][s][l] = 0.0;
                        }

                        if (right_sums[k][s][l] < 0.0) {
                            right_sums[k][s][l] = 0.0;
                        }

                        left_sums[k][s][l] = std::round(left_sums[k][s][l]);
                        right_sums[k][s][l] = std::round(right_sums[k][s][l]);
                    }
                }
            }

            std::vector<std::vector<double>> total_left_sums(access_features(data[0]).size(), std::vector<double>(splits.size(), 0.0));
            std::vector<std::vector<double>> total_right_sums(access_features(data[0]).size(), std::vector<double>(splits.size(), 0.0));

            for (size_t k = 0; k < access_features(data[0]).size(); k++) {
                for (size_t l = 0; l < access_labels(data[0]).size(); l++) {
                    for (size_t s = 0; s < splits.size(); s++) {
                        total_left_sums[k][s] += left_sums[k][s][l];
                        total_right_sums[k][s] += right_sums[k][s][l];
                    }
                }
            }

            double epsilon = std::numeric_limits<double>::epsilon();

            std::vector<std::vector<double>> gini(access_features(data[0]).size(), std::vector<double>(splits.size(), 0.0));
            for (size_t k = 0; k < access_features(data[0]).size(); k++) {
                for (size_t s = 0; s < splits.size(); s++) {

                    double squared_fraction_labels_sum_left = 0.0;
                    double squared_fraction_labels_sum_right = 0.0;

                    if (total_left_sums[k][s] > epsilon) {
                        for (size_t l = 0; l < access_labels(data[0]).size(); l++) {
                            squared_fraction_labels_sum_left +=  std::pow(left_sums[k][s][l] / total_left_sums[k][s], 2);
                        }

                        gini[k][s] = (1.0 - squared_fraction_labels_sum_left) * total_left_sums[k][s];
                    }


                    if (total_right_sums[k][s] > epsilon) {
                        for (size_t l = 0; l < access_labels(data[0]).size(); l++) {
                            squared_fraction_labels_sum_right +=  std::pow(right_sums[k][s][l] / total_right_sums[k][s], 2);
                        }

                        gini[k][s] += (1.0 - squared_fraction_labels_sum_right) * total_right_sums[k][s];
                    }
                }
            }

            size_t min_feature_index = 0, min_split_index = 0;
            double min_gini = std::numeric_limits<double>::max();
            for (size_t k = 0; k < access_features(data[0]).size(); k++) {
                for (size_t s = 0; s < splits.size(); s++) {
                    if (gini[k][s] < min_gini) {
                        min_feature_index = k;
                        min_split_index = s;
                        min_gini = gini[k][s];
                    }
                }
            }

            tree[node_count] = make_shared<decision_tree_node>(node_count,min_feature_index, splits[min_split_index]);

            //Step 3
            std::vector<double> weights_right(weights.size());
            for (size_t i = 0; i < weights.size(); i++) {
                weights_right[i] = weights[i] * soft_ifs[i][min_feature_index][min_split_index];
                weights[i] = weights[i] - weights_right[i];
            }

            build_base_tree(tree, data, weights, splits, height - 1, soft_if_poly_degree, 2 * node_count + 1);   //left
            build_base_tree(tree, data, weights_right, splits, height - 1, soft_if_poly_degree, 2 * node_count + 2);     //right
        }
    }

    static vector<shared_ptr<decision_tree_node>> train_plain_tree(const string& plain_file_full_path, size_t features_count,
                                                            size_t labels_count, size_t tree_height, int soft_if_poly_degree, double split_step_size = 0.05) {
        vector<Sample<double>> data = read_data(plain_file_full_path, features_count, labels_count);
        vector<double> weights(data.size(), 1.0);
        double begin_value = -1.0, end_value  = 1.0;
        size_t splits_count = ceil((end_value - begin_value) / split_step_size) + 1;
        vector<double> splits(splits_count);
        double current_value = begin_value - split_step_size;
        std::generate_n(splits.begin(), splits_count, [&] {  current_value += split_step_size; return current_value;} );

        size_t nodes_count = pow(2, tree_height + 1) - 1;
        size_t non_leaf_count = nodes_count - pow(2, tree_height);
        vector<shared_ptr<decision_tree_node>> tree(nodes_count);
        build_base_tree(tree, data, weights, splits, tree_height, soft_if_poly_degree);
        for (size_t i = 0; i < non_leaf_count; i++) {
            tree[i]->set_left_node(tree[(2 * i) + 1]);
            tree[i]->set_right_node(tree[(2 * i) + 2]);
        }

        return tree;
    }

    static vector<shared_ptr<decision_tree_node>> make_delta_tree(vector<shared_ptr<decision_tree_node>> tree1, vector<shared_ptr<decision_tree_node>> tree2, int tree_height) {
        vector<shared_ptr<decision_tree_node>> delta(tree1.size());
        size_t nodes_count = pow(2, tree_height + 1) - 1;
        size_t non_leaf_count = nodes_count - pow(2, tree_height);

        for (size_t i = 0; i < nodes_count; i++) {
            if (tree1[i]->is_leaf()) {
                vector<double> values1 = tree1[i]->get_leaf_value();
                vector<double> values2 = tree2[i]->get_leaf_value();
                vector<double> leaf_values(values1.size(), 0.0);

                for (size_t j = 0; j < leaf_values.size(); j++) {
                    leaf_values[j] = abs(values1[j] - values2[j]);
                }

                delta[i] = make_shared<decision_tree_node>(i, leaf_values);
            }
            else {
                int feature1 = tree1[i]->get_feature_index();
                int feature2 = tree2[i]->get_feature_index();
                double theta1 = tree1[i]->get_theta();
                double theta2 = tree2[i]->get_theta();
                int delta_feature = abs(feature1 - feature2);
                double delta_theta = abs(theta1 - theta2);

                delta[i] = make_shared<decision_tree_node>(i,delta_feature, delta_theta);
            }
        }

        for (size_t i = 0; i < non_leaf_count; i++) {
            delta[i]->set_left_node(delta[(2 * i) + 1]);
            delta[i]->set_right_node(delta[(2 * i) + 2]);
        }

        return delta;
    }
};

/**
 *
 * @b Description:
 * Test fixture class for soft-if polynomial evaluation over encrypted data.
 */
class SoftIf : public ::testing::TestWithParam<int> {
protected:
    const size_t _instance_count = 20;
    SEAL_c _seal;
    vector<double> _x;
    vector<Ciphertext>_cipher_x;

    SoftIf() {
        switch(GetParam()) {
            case 32:
                _seal = {55, 15, { 60, 55, 55, 55, 55, 55, 55, 55, 55, 55, 60 }};
                break;
            case 16:
                _seal = {40, 14, { 50, 40, 40, 40, 40, 40, 40, 40, 40, 50 }};
                break;
            default:
                _seal = {30, 14, { 50, 30, 30, 30, 30, 30, 30, 30, 50 }};
                break;
        }
        _seal.generate_keys();
        vector<double> x(_instance_count);
        vector<Plaintext> plain_x(_instance_count);
        vector<Ciphertext> cipher_x(_instance_count);
        default_random_engine gen(1);
        uniform_real_distribution<double> real_dist(-2.0,2.0);

        for (size_t i = 0; i < x.size(); i++) {
            x[i] = real_dist(gen);

            _seal.encoder_ptr->encode(x[i], _seal.scale, plain_x[i]);
            _seal.encryptor_ptr->encrypt(plain_x[i], cipher_x[i]);
        }

        _x = x;
        _cipher_x = cipher_x;
    }

    [[nodiscard]] vector<double> generate_delta(const soft_if& soft_if_op, vector<double> x,
                                                vector<Ciphertext> cipher_x) const{
        vector<double> delta(x.size());
        double plain_result;
        Ciphertext ciphered_result;
        Plaintext decrypted_result;
        vector<double> soft_if_result;

        for (size_t i = 0; i < x.size(); i++) {
            plain_result = PlainMethods::plain_soft_if(x[i], soft_if_op.getCoeffs(), soft_if_op.getFreeCoeff());
            soft_if_op(ciphered_result, cipher_x[i]);
            _seal.decryptor_ptr->decrypt(ciphered_result, decrypted_result);
            _seal.encoder_ptr->decode(decrypted_result, soft_if_result);
            delta[i] = abs(soft_if_result[0] - plain_result);
        }

        return delta;
    }
};

/**
 *
 * @b Description:
 * Test fixture class for tree evaluation evaluation over encrypted data.
 */
class TreeEval : public ::testing::TestWithParam<int> {
protected:
    const size_t _features_count = 46;
    vector<shared_ptr<decision_tree_node>> _nodes;
    SEAL_c _seal;
    vector<double> _x;
    vector<Ciphertext>_cipher_x;

    TreeEval() {
        _nodes = generateSeededTree(_features_count, 4);
        switch(GetParam()) {
            case 32:
                _seal = {45, 15, { 60, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 60 }};
                break;
            case 16:
                _seal = {35, 14, { 50, 35, 35, 35, 35, 35, 35, 35, 35, 35, 50 }};
                break;
            default:
                _seal = {30, 14, { 50, 30, 30 ,30, 30, 30, 30, 30, 30, 50 }};
                break;
        }
        _seal.generate_keys();
        vector<double> x(_features_count);
        vector<Plaintext> plain_x(_features_count);
        vector<Ciphertext> cipher_x(_features_count);
        default_random_engine gen(1);
        uniform_real_distribution<double> real_dist(-1.0,1.0);

        for (size_t i = 0; i < x.size(); i++) {
            x[i] = real_dist(gen);

            _seal.encoder_ptr->encode(x[i], _seal.scale, plain_x[i]);
            _seal.encryptor_ptr->encrypt(plain_x[i], cipher_x[i]);
        }

        _x = x;
        _cipher_x = cipher_x;
    }

    static vector<shared_ptr<decision_tree_node>> generateSeededTree(int features_count, int tree_height){
        size_t nodes_count = pow(2, tree_height + 1) - 1;
        size_t non_leaf_count = nodes_count - pow(2, tree_height);
        vector<shared_ptr<decision_tree_node>> nodes (nodes_count);

        unsigned seed = 1;
        default_random_engine gen(seed);
        uniform_real_distribution<double> real_dist(-1.0,1.0);
        uniform_int_distribution<int> int_dist(0,features_count-1);

        for (size_t i = 0; i < non_leaf_count; i++) {
            nodes[i] = make_shared<decision_tree_node>(i,int_dist(gen), real_dist(gen));
        }

        for (size_t i = non_leaf_count; i < nodes_count; i++) {
            vector<double> v = {real_dist(gen), real_dist(gen)};
            nodes[i] = make_shared<decision_tree_node>(i, v);
        }

        for (size_t i = 0; i < non_leaf_count; i++) {
            nodes[i]->set_left_node(nodes[(2 * i) + 1]);
            nodes[i]->set_right_node(nodes[(2 * i) + 2]);
        }

        return nodes;
    }

    static vector<double> tree_eval_plain(decision_tree_node& node, vector<double>& x, soft_if soft_if_op) {
        if (node.is_leaf()) {
            return node.get_leaf_value();
        }
        else {
            double x_minus_theta = x[node.get_feature_index()] - node.get_theta();
            double soft_if_result = PlainMethods::plain_soft_if(x_minus_theta, soft_if_op.getCoeffs(), soft_if_op.getFreeCoeff());


            vector<double> result;
            if (node.has_right_node() && node.has_left_node()){
                vector<double> right_sub_tree = tree_eval_plain(*node.get_right_node(), x, soft_if_op);
                vector<double> left_sub_tree = tree_eval_plain(*node.get_left_node(), x, soft_if_op);
                if (size(right_sub_tree) == size(left_sub_tree) && size(right_sub_tree) > 0){
                    for (int i=0; i<size(right_sub_tree); i++){
                        result.push_back(soft_if_result*right_sub_tree[i] + (1.0-soft_if_result)*left_sub_tree[i]);
                    }
                }
                else {
                    throw invalid_argument( "all leaves must have exactly the same (>0) number of labels" );
                }
            }
            else{
                throw invalid_argument( "non-leaf node must have exactly two children" );
            }
            return result;
        }
    }

    vector<double> generate_delta(tree_eval_server tree_eval, vector<double> eval_result_plain, Ciphertext eval_result_cipher){
        Plaintext plain_result;
        vector<double> result;
        _seal.decryptor_ptr->decrypt(eval_result_cipher, plain_result);
        _seal.encoder_ptr->decode(plain_result, result);

        vector<double> delta(result.size());
        for (size_t i = 0; i < result.size();i++) {
            delta[i] = abs(result[i] - eval_result_plain[i]);
        }

        return delta;
    }
};

/**
 *
 * @b Description:
 * Test fixture class for training a tree over encrypted data.
 */
class TreeTrain : public ::testing::TestWithParam<int> {
protected:
    string _plain_file_full_path = "../data/default_data.csv";
    int _features_count = 2;
    int _labels_count = 2;

    int _poly_modulus_degree;
    int _scale;
    vector<int> _bit_sizes;
    int _soft_if_poly_degree;

    double _split_step_size = 0.5;
    int _tree_height = 4;

    TreeTrain() {
        _poly_modulus_degree = 15;
        _scale = 50;
        _bit_sizes = { 60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 60 };
        _soft_if_poly_degree = GetParam();
    }
};

/**
 *
 * @b Description:
 * Test soft-if polynomial evaluation over encrypted data. This test assumes that PlainMethods::soft_if_sparse() is correct.
 * It compares the result of evaluating the polynomial over encrypted data to the result of evaluating over plain data.
 * The differences between the values in the two result vectors should be within the expected noise
 *
 */
TEST_P(SoftIf, differentPolyDegrees) {
    soft_if soft_if_op(_seal, GetParam());

    vector<double> delta = generate_delta(soft_if_op, _x, _cipher_x);

    for (size_t i = 0; i < delta.size(); i++) {
        EXPECT_LE(delta[i], 0.01) << "where x value: " << _x[i] <<
        ", delta between soft_if result and plain soft_if result too high.";
    }
}

INSTANTIATE_TEST_SUITE_P(PredictionTest, SoftIf, testing::Values(8, 16, 32));

/**
 *
 * @b Description:
 * Test tree evaluation over encrypted data. This test assumes that tree_eval_plain() is correct.
 * It compares the result of evaluating the tree over encrypted data to the result of evaluating over plain data.
 * The differences between the values in the two result vectors should be within the expected noise
 *
 */
 TEST_P(TreeEval, differentPolyDegrees) {
    soft_if soft_if_op(_seal, GetParam());
    tree_eval_server tree_eval(_seal, GetParam());
    vector<double> eval_result_plain = tree_eval_plain(*_nodes[0], _x, soft_if_op);
    Ciphertext eval_result_cipher = tree_eval.predict(*_nodes[0], _cipher_x);
    vector<double> delta = generate_delta(tree_eval, eval_result_plain, eval_result_cipher);

    for (size_t i = 0; i < eval_result_plain.size();i++) {
        EXPECT_LE(delta[i], 0.01) <<
        "Delta between tree_eval result and tree_eval_plain result too high.";
    }
}

INSTANTIATE_TEST_SUITE_P(PredictionTest, TreeEval, testing::Values(8, 16, 32));

/**
 *
 * @b Description:
 * Test tree training over encrypted data. This test assumes that PlainMethods::train_plain_tree() is correct.
 * It compares the result of training over encrypted data to the result of training over plain data. The differences between
 * the values on the leaf nodes when comparing the two trees should be within the expected noise
 *
 */
 TEST_P(TreeTrain, differentPolyDegrees) {
    fhe_client client(_scale, _poly_modulus_degree, _bit_sizes, _split_step_size);
    vector<Sample<Ciphertext>> enc_data = client.get_encrypted_data(_plain_file_full_path, _features_count, _labels_count);
    PublicKeys client_public_keys = client.get_public_keys();
    stringstream data_stream;

    tree_train_server server(_scale, _poly_modulus_degree, _bit_sizes,
                             client_public_keys, client,
                             _features_count, _labels_count, _tree_height,
                             _soft_if_poly_degree, enc_data, _split_step_size);
    server.execute(data_stream);

    vector<shared_ptr<decision_tree_node>> dec_trained_tree = client.decrypt_and_save_tree(data_stream, "");
    vector<shared_ptr<decision_tree_node>> plain_tree = PlainMethods::train_plain_tree(_plain_file_full_path, _features_count, _labels_count, _tree_height, _soft_if_poly_degree, _split_step_size);
    vector<shared_ptr<decision_tree_node>> delta_tree = PlainMethods::make_delta_tree(dec_trained_tree, plain_tree, _tree_height);

    for (auto & i : delta_tree) {
        if (i->is_leaf()) {
            for (size_t j = 0; j < i->get_leaf_value().size(); j++) {
                EXPECT_LE(i->get_leaf_value()[j], 0.02) <<
                "Delta between client-decrypted tree leaf and plain tree leaf too high.";
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TrainingTest, TreeTrain, testing::Values(8, 16, 32));

/**
*
* @b Description:
* Test client's gini_based_min_feature_split(), using randomly generated (set seed) node_sums_arr in python and read by
 * PlainMethods::read_gini_test_vector() (the vector can be found in gini_tet_vector.txt).
 * the test compares the client's results vs plain python generated results (generated in the provided python code gini_testing.py).
*
*/
TEST(GiniTest, PyCompare) {
    int labels_count = 3;
    int features_count = 5;
    int thetas_count = 40;
    vector<vector<int>> ranges{{0, 400}, {0, 400}, {0, 400}, {0, 400}, {0, 400}, {0, 400}, {0, 400}, {0, 400}, {-1, 0}};
    vector<vector<double>> py_results{{1, 6}, {1, 36}, {0, 37}, {3, 0}, {4, 14}, {1, 34}, {3, 38}, {3, 12}, {0, 0}, {0, 0}};
    string line;
    vector<pair<vector_3d<double>, vector_3d<double>>> node_sums_arr = PlainMethods::read_gini_test_vector();

    for (size_t i = 0; i < 10; i++) {
        fhe_client client{50, 15, { 60, 50, 60 }};
        client._features_count = features_count;
        client._labels_count = labels_count;
        client._num_of_splits = thetas_count;
        pair<int,int> result = client.gini_based_min_feature_split(node_sums_arr[i].first, node_sums_arr[i].second);

        double expected_feature = py_results[i][0], expected_threshold = py_results[i][1];
        int res_feature = result.first, res_threshold = result.second;
        EXPECT_TRUE((expected_feature == res_feature) && (expected_threshold == res_threshold)) <<
        "Where gini recieved values in range [" << ranges[i][0] << ", " << ranges[i][1] << "]:" << endl <<
        "py result -> F: " << expected_feature << " T: " << expected_threshold << endl <<
        "c++ result -> F: " << res_feature << " T: " << res_threshold << endl;
    }
}
