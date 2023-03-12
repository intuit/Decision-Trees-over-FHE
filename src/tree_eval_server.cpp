#include "tree_eval_server.h"
#include "utils.h"

Ciphertext tree_eval_server::predict(decision_tree_node& node, std::vector<Ciphertext>& x) {
    //Recursively traverse the tree and using soft-if to calculate the encrpyted prediction.
    if (node.is_leaf()) {
        //Encrypt the value of the leaf
        Plaintext plain_value;
        _seal.encoder_ptr->encode(node.get_leaf_value(), _seal.scale, plain_value);
        Ciphertext ciphertext_value;
        _seal.encryptor_ptr->encrypt(plain_value, ciphertext_value);

        return ciphertext_value;
    }
    else if (node.has_right_node() && node.has_left_node()) {
        //Calculate (x[feature] - theta)
        Ciphertext substruction_result;
        Plaintext plain_theta;
        _seal.encoder_ptr->encode(node.get_theta(), _seal.scale, plain_theta);
        _seal.evaluator_ptr->sub_plain(x[node.get_feature_index()], plain_theta, substruction_result);

        Ciphertext soft_if_result;
        _soft_if_op(soft_if_result, substruction_result);

        //Recursively calculate right side
        Ciphertext right_sub_tree = predict(*node.get_right_node(), x);
        _seal.align_modulus(soft_if_result, right_sub_tree);

        //right * soft_if_res
        _seal.evaluator_ptr->multiply_inplace(right_sub_tree, soft_if_result);
        _seal.evaluator_ptr->relinearize_inplace(right_sub_tree, *_seal.relin_keys);
        _seal.evaluator_ptr->rescale_to_next_inplace(right_sub_tree);

        //Recursively calculate left side
        Ciphertext left_sub_tree = predict(*node.get_left_node(), x);

        //Left * (1 - soft_if_res)
        Plaintext plain_one;
        _seal.encoder_ptr->encode(1.0, soft_if_result.parms_id(), soft_if_result.scale(), plain_one);
        Ciphertext one_minus_soft_if_result;
        _seal.evaluator_ptr->sub_plain(soft_if_result, plain_one, one_minus_soft_if_result);
        _seal.evaluator_ptr->negate_inplace(one_minus_soft_if_result);
        _seal.align_modulus(one_minus_soft_if_result, left_sub_tree);
        _seal.evaluator_ptr->multiply_inplace(left_sub_tree, one_minus_soft_if_result);
        _seal.evaluator_ptr->relinearize_inplace(left_sub_tree, *_seal.relin_keys);
        _seal.evaluator_ptr->rescale_to_next_inplace(left_sub_tree);

        //Right + Left
        _seal.align_modulus(right_sub_tree, left_sub_tree);
        right_sub_tree.scale() = left_sub_tree.scale();
        _seal.evaluator_ptr->add_inplace(right_sub_tree, left_sub_tree);

        return right_sub_tree;
    }
// this is reached in case the tree is not binary, this is considered an invalid decision tree
    return x[0];
}

vector<shared_ptr<decision_tree_node>> tree_eval_server::load_plain_tree(const string& source) {
    ifstream i_file(source);
    if (i_file.is_open())
    {
        size_t tree_height, labels_count, features_count;
        i_file >> tree_height;
        i_file >> labels_count;
        i_file >> features_count;

        size_t nodes_count = pow(2, tree_height + 1) - 1;
        size_t non_leaf_count = nodes_count - pow(2, tree_height);
        vector<shared_ptr<decision_tree_node>> nodes (nodes_count);

        for (size_t index = 0; index < non_leaf_count; index++) {
            int feature_index;
            double theta;
            i_file >> feature_index;
            i_file >> theta;
            nodes[index] = make_shared<decision_tree_node>(index,feature_index,theta);
        }
        for (size_t index = non_leaf_count; index < nodes_count; index++) {
            vector<double> values(size_t(labels_count), 0);
            for (int i = 0; i < labels_count; i++) {
                i_file >> values[i];
            }
            nodes[index] = make_shared<decision_tree_node>(index,values);
        }
        i_file.close();

        for (size_t i = 0; i < non_leaf_count; i++) {
            nodes[i]->set_left_node(nodes[(2 * i) + 1]);
            nodes[i]->set_right_node(nodes[(2 * i) + 2]);
        }

        return nodes;
    } else throw invalid_argument("Unable to read tree files!");
}

Ciphertext tree_eval_server::execute(const string& source, std::vector<Ciphertext>& x) {
    vector<shared_ptr<decision_tree_node>> nodes = load_plain_tree(source);
    return predict(*nodes[0], x);
}
