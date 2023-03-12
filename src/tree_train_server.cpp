#include "tree_train_server.h"
#include "enc_tree_node.h"
#include <utility>

TrainingData<Ciphertext> tree_train_server::prepare_training_data() {
    // Initialize the size of the training data (chunks * features * splits * labels)
    size_t chunks = _data.size();
    TrainingData<Ciphertext> training_data;
    training_data_init<Ciphertext>(training_data,
                                   chunks, _features_count, _splits.size(), _labels_count);

    // Generate a plaintext of 1
    Plaintext plain_one;
    _seal.encoder_ptr->encode(1.0, _seal.scale, plain_one);

    // Generate the training data from the context's data
    for (size_t i = 0; i < chunks; i++) {
        Sample<Ciphertext>& sample = _data[i];

        for (size_t k = 0; k < _features_count; k++) {
            for (size_t s = 0; s < _splits.size(); s++) {
                Ciphertext& soft_if_right_ref = access_soft_ifs_right<Ciphertext>(training_data)[i][k][s];
                Ciphertext& soft_if_left_ref = access_soft_ifs_left<Ciphertext>(training_data)[i][k][s];

                // soft_if_right = soft_if(x[i][k] - splits[s])
                Ciphertext& x_feature_ref = access_features(sample)[k];
                _seal.evaluator_ptr->sub_plain(x_feature_ref, _splits[s], soft_if_right_ref);
                soft_if soft_if_op(_seal, (int)_soft_if_degree);
                soft_if_op(soft_if_right_ref);

                // soft_if_left = negate(soft_if_right - 1) = 1 - soft_if_right
                _seal.evaluator_ptr->mod_switch_to_inplace(plain_one, soft_if_right_ref.parms_id());
                plain_one.scale() = soft_if_right_ref.scale();
                _seal.evaluator_ptr->sub_plain(soft_if_right_ref, plain_one, soft_if_left_ref);
                _seal.evaluator_ptr->negate_inplace(soft_if_left_ref);

                for (size_t l = 0; l < _labels_count; l++) {
                    Ciphertext& labeled_soft_if_right_ref = access_labeled_soft_ifs_right<Ciphertext>(training_data)[i][k][s][l];
                    Ciphertext& labeled_soft_if_left_ref = access_labeled_soft_ifs_left<Ciphertext>(training_data)[i][k][s][l];

                    // labeled_soft_if_right_ref = y[l] * soft_if_right
                    Ciphertext& y_label_ref  = access_labels(sample)[l];
                    _seal.evaluator_ptr->mod_switch_to(y_label_ref, soft_if_right_ref.parms_id(), labeled_soft_if_right_ref);
                    labeled_soft_if_right_ref.scale() = soft_if_right_ref.scale();
                    _seal.evaluator_ptr->multiply_inplace(labeled_soft_if_right_ref , soft_if_right_ref);
                    _seal.evaluator_ptr->relinearize_inplace(labeled_soft_if_right_ref , *_seal.relin_keys);
                    _seal.evaluator_ptr->rescale_to_next_inplace(labeled_soft_if_right_ref );

                    // labeled_soft_if_left_ref = y[l] * (1 - soft_if_right) = y[l] - labeled_soft_if_right_ref
                    Ciphertext y_label;
                    _seal.evaluator_ptr->mod_switch_to(y_label_ref, labeled_soft_if_right_ref.parms_id(), y_label);
                    y_label.scale() = labeled_soft_if_right_ref.scale();
                    _seal.evaluator_ptr->sub(y_label, labeled_soft_if_right_ref , labeled_soft_if_left_ref);
                }
            }
        }
    }

    return training_data;
}

vector_3d<Ciphertext> tree_train_server::weighted_sum_by_feature_split_label(const vector<Ciphertext>& weights, const vector_4d<Ciphertext>& labeled_soft_ifs) {
    vector_3d<Ciphertext> node_sums(_features_count,
                                    vector_2d<Ciphertext>(_splits.size(),
                                                          vector<Ciphertext>(_labels_count)));
    size_t chunks = labeled_soft_ifs.size();

    for (size_t k = 0; k < _features_count; k++) {
        for (size_t s = 0; s < _splits.size(); s++) {
            for (size_t l = 0; l < _labels_count; l++) {
                for (size_t i = 0; i < chunks; i++ ) {
                    Ciphertext mod_switched;
                    const Ciphertext& mod_not_switched = _seal.align_modulus_to(weights[i], labeled_soft_ifs[i][k][s][l], mod_switched);
                    _seal.evaluator_ptr->multiply_inplace(mod_switched, mod_not_switched);
                    _seal.evaluator_ptr->relinearize_inplace(mod_switched , *_seal.relin_keys);
                    _seal.evaluator_ptr->rescale_to_next_inplace(mod_switched);

                    if (i == 0) {
                        node_sums[k][s][l] = mod_switched; // copy assignment
                    }
                    else {
                        _seal.evaluator_ptr->add_inplace(node_sums[k][s][l], mod_switched);
                    }
                }
            }
        }
    }

    return node_sums;
}

vector<Ciphertext> tree_train_server::update_weights(const vector_3d<Ciphertext>& soft_ifs, const vector_2d<Ciphertext>& one_hot_selector,
                                                     const vector<Ciphertext>& parent_weights, size_t chunks, bool is_root) {
    vector<Ciphertext> updated_weights(chunks);

    for (size_t i = 0; i < chunks; i++) {
        Ciphertext& updated_weight = updated_weights[i];
        _seal.encryptor_ptr->encrypt_zero(updated_weight);

        for (size_t k = 0; k < _features_count; k++) {
            for (size_t s = 0; s < _splits.size(); s++) {
                Ciphertext temp;
                _seal.evaluator_ptr->mod_switch_to(one_hot_selector[k][s], soft_ifs[i][k][s].parms_id(), temp);
                temp.scale() = soft_ifs[i][k][s].scale();
                _seal.evaluator_ptr->multiply_inplace(temp, soft_ifs[i][k][s]);
                _seal.evaluator_ptr->relinearize_inplace(temp , *_seal.relin_keys);
                _seal.evaluator_ptr->rescale_to_next_inplace(temp);

                _seal.evaluator_ptr->mod_switch_to_inplace(updated_weight, temp.parms_id());
                updated_weight.scale() = temp.scale();
                _seal.evaluator_ptr->add_inplace(updated_weight, temp);
            }
        }

        if (!is_root) {
            Ciphertext mod_switched;
            const Ciphertext& mod_not_switched = _seal.align_modulus_to(parent_weights[i], updated_weight, mod_switched);
            _seal.evaluator_ptr->multiply_inplace(mod_switched, mod_not_switched);
            _seal.evaluator_ptr->relinearize_inplace(mod_switched , *_seal.relin_keys);
            _seal.evaluator_ptr->rescale_to_next_inplace(mod_switched);

            updated_weight = mod_switched;
        }
    }

    return updated_weights;
}

vector<Ciphertext> tree_train_server::process_leaf(queue<vector<Ciphertext>>& weights_by_node_queue) {
    vector<Ciphertext>& parent_weights = weights_by_node_queue.front();
    vector<Ciphertext> label_scores(_labels_count);

    for (size_t l = 0; l < _labels_count; l++) {
        _seal.encryptor_ptr->encrypt_zero(label_scores[l]);

        for (size_t i = 0; i < _data.size(); i++) {
            Ciphertext temp;
            _seal.evaluator_ptr->mod_switch_to(access_labels(_data[i])[l], parent_weights[i].parms_id() ,temp);
            temp.scale() = parent_weights[i].scale();
            _seal.evaluator_ptr->multiply_inplace(temp, parent_weights[i]);
            _seal.evaluator_ptr->relinearize_inplace(temp , *_seal.relin_keys);
            _seal.evaluator_ptr->rescale_to_next_inplace(temp);

            _seal.evaluator_ptr->mod_switch_to_inplace(label_scores[l], temp.parms_id());
            label_scores[l].scale() = temp.scale();
            _seal.evaluator_ptr->add_inplace(label_scores[l], temp);
        }
    }

    weights_by_node_queue.pop();

    return label_scores;
}

pair<Ciphertext, Ciphertext> tree_train_server::process_node(queue<vector<Ciphertext>>& weights_by_node_queue, TrainingData<Ciphertext>& training_data, size_t chunks, size_t index) {
    vector<Ciphertext> parent_weights = weights_by_node_queue.front();

    // Over encrypted, generate each side's weighted sum of ((x - split) * label).
    vector_3d<Ciphertext> node_sums_left = weighted_sum_by_feature_split_label(parent_weights, access_labeled_soft_ifs_left<Ciphertext>(training_data));
    vector_3d<Ciphertext> node_sums_right = weighted_sum_by_feature_split_label(parent_weights, access_labeled_soft_ifs_right<Ciphertext>(training_data));

    // Over plain in the client, generate the node content (one hot selector, feature, split) using gini-index
    // based on the calculated weighted sums.
    vector_2d<Ciphertext> one_hot_selector;
    Ciphertext feature, split;
    tie(one_hot_selector, feature, split) = _client.generate_node_content(node_sums_left, node_sums_right); // Client's public function.

    // Over encrypted, update the weights queue with the children weights.
    vector<Ciphertext> weights_left = update_weights(access_soft_ifs_left<Ciphertext>(training_data), one_hot_selector, parent_weights, chunks, index == 0);
    weights_by_node_queue.push(weights_left);

    vector<Ciphertext> weights_right = update_weights(access_soft_ifs_right<Ciphertext>(training_data), one_hot_selector, parent_weights, chunks, index == 0);
    weights_by_node_queue.push(weights_right);

    weights_by_node_queue.pop();

    pair<Ciphertext, Ciphertext> res(feature, split);
    return res;
}

void tree_train_server::execute(stringstream& datastream) {
    // Initialize the training data and the tree.
    TrainingData<Ciphertext> training_data = prepare_training_data();
    queue<vector<Ciphertext>> weights_by_node_queue;
    size_t nodes_count = (size_t)pow(2, _tree_height + 1) - 1;
    vector<shared_ptr<enc_tree_node>> nodes(nodes_count);

    // Push the starting root weights as 1's to the queue.
    Plaintext plain_one;
    Ciphertext enc_one;
    _seal.encoder_ptr->encode(1.0, _seal.scale, plain_one);
    _seal.encryptor_ptr->encrypt(plain_one, enc_one);
    vector<Ciphertext> root_weights(_data.size(), enc_one);
    weights_by_node_queue.push(root_weights);

    // Train all nodes
    for (size_t node_index = 0; node_index < nodes_count; node_index++) {
        if (node_index < nodes_count / 2) { //non-leaf
            pair<Ciphertext, Ciphertext> node_values = process_node(weights_by_node_queue, training_data, _data.size(), node_index);
            nodes[node_index] = make_shared<enc_tree_node>(node_index,node_values.first, node_values.second);
        }
        else if (node_index >= nodes_count / 2){ //leaf
            vector<Ciphertext> leaf_values = process_leaf(weights_by_node_queue);
            nodes[node_index] = make_shared<enc_tree_node>(node_index, leaf_values);
        }
    }

    stream_tree(nodes, datastream);
}

void tree_train_server::stream_tree(const vector<shared_ptr<enc_tree_node>>& tree, stringstream& destination) const {
    size_t nodes_count = (size_t)pow(2, _tree_height + 1) - 1;
    size_t non_leaf_count = nodes_count - (size_t)pow(2, _tree_height);

    destination.write(reinterpret_cast<char const*>(&_tree_height), sizeof(_tree_height));
    destination.write(reinterpret_cast<char const*>(&_labels_count), sizeof(_labels_count));
    destination.write(reinterpret_cast<char const*>(&_features_count), sizeof(_features_count));

    for (size_t i = 0; i < non_leaf_count; i++) {
        tree[i]->get_feature_index().save(destination);
        tree[i]->get_theta().save(destination);
    }

    for (size_t i = non_leaf_count; i < nodes_count; i++) {
        for (int j = 0; j < _labels_count; j++) {
            tree[i]->get_leaf_value()[j].save(destination);
        }
    }
}
