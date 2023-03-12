#include "fhe_client.h"

vector<double> fhe_client::encode_splits() const {
    vector<double> splits_doubles(_num_of_splits);
    double current_value = _split_initial_value;

    for (size_t s = 0; s < _num_of_splits; s++) {
        splits_doubles[s] = current_value;
        current_value += _split_step_size;
    }

    return splits_doubles;
}

vector<Sample<Ciphertext>> fhe_client::get_encrypted_data(string& plain_file_full_path, size_t features_count, size_t labels_count) {
    _features_count = features_count;
    _labels_count = labels_count;

    vector<Sample<double>> plain_data = read_data(plain_file_full_path, features_count, labels_count);

    _samples = plain_data.size();
    _chunks = ceil(_samples / _seal.slot_count) + 1;

    _enc_data = encrypt_packed_samples(plain_data, _chunks);

    plain_data.clear();

    return _enc_data;
}

vector<Sample<Ciphertext>> fhe_client::encrypt_packed_samples(vector<Sample<double>>& plain_data, size_t chunks)  {
    vector<Sample<Ciphertext>> enc_data(chunks);

    // The data is divided into chunks, where each chunk contains a ciphertext per feature, plus a ciphertext per label index.
    // For m features and n labels, we have m + n ciphertexts per chunk.
    for (size_t c = 0; c < enc_data.size(); c++) {
        sample_init(enc_data[c], _features_count, _labels_count);

        for (size_t k = 0; k < _features_count; k++) {
            vector<double> x_chunk(_seal.slot_count, 0.0);
            for (size_t i = 0; i < _seal.slot_count && ((c * _seal.slot_count) + i) < plain_data.size(); i++) {
                x_chunk[i] = access_features(plain_data[(c * _seal.slot_count) + i])[k];
            }

            Plaintext ptxt_x_chunk;
            _seal.encoder_ptr->encode(x_chunk, _seal.scale, ptxt_x_chunk);
            _seal.encryptor_ptr->encrypt(ptxt_x_chunk, access_features(enc_data[c])[k]);
        }

        for (size_t l = 0; l < _labels_count; l++) {
            vector<double> y_chunk(_seal.slot_count, 0.0);
            for (size_t i = 0; i < _seal.slot_count && ((c * _seal.slot_count) + i) < plain_data.size(); i++) {
                y_chunk[i] = access_labels(plain_data[(c * _seal.slot_count) + i])[l];
            }

            Plaintext ptxt_y_chunk;
            _seal.encoder_ptr->encode(y_chunk, _seal.scale, ptxt_y_chunk);
            _seal.encryptor_ptr->encrypt(ptxt_y_chunk, access_labels(enc_data[c])[l]);
        }
    }

    return enc_data;
}

vector_3d<double> fhe_client::dec_sums_by_feature_split_label(const vector_3d<Ciphertext>& enc_sums) {
    vector_3d<double> plain_sums(_features_count,
                                 vector_2d<double>(_num_of_splits,
                                                   vector<double>(_labels_count)));

    for (size_t k = 0; k < _features_count; k++) {
        for (size_t s = 0; s < _num_of_splits; s++) {
            for (size_t l = 0; l < _labels_count; l++) {
                Plaintext ptxt;
                vector<double> ptxt_vector;
                _seal.decryptor_ptr->decrypt(enc_sums[k][s][l], ptxt);
                _seal.encoder_ptr->decode(ptxt, ptxt_vector);
                plain_sums[k][s][l] = accumulate(ptxt_vector.begin(), ptxt_vector.end(), 0.0);
            }
        }
    }

    return plain_sums;
}

void fhe_client::smooth_sums(vector_3d<double>& sums) {
    for (size_t k = 0; k < _features_count; k++) {
        for (size_t s = 0; s < _num_of_splits; s++) {
            for (size_t l = 0; l < _labels_count; l++) {
                if (sums[k][s][l] < 0.0) {
                    sums[k][s][l] = 0.0;
                }

                sums[k][s][l] = round(sums[k][s][l]);
            }
        }
    }
}

void fhe_client::add_single_side_to_gini(const vector_3d<double>& sums, vector_2d<double>& gini) {
    vector_2d<double> total_sums(_features_count, vector<double>(_num_of_splits,0.0));

    // Sum all labels for each feature and split (calculate total_side)
    for (size_t k = 0; k < _features_count; k++) {
        for (size_t s = 0; s < _num_of_splits; s++) {
            for (size_t l = 0; l < _labels_count; l++) {
                total_sums[k][s] += sums[k][s][l];
            }
        }
    }

    // Calculate and add the Gini impurity ( (side[l]/total_side)^2 )
    for (size_t k = 0; k < _features_count; k++) {
        for (size_t s = 0; s < _num_of_splits; s++) {
            double squared_fraction_all_labels = 0.0;
            if (total_sums[k][s] > 0) {
                for (size_t l = 0; l < _labels_count; l++) {
                    squared_fraction_all_labels += pow(sums[k][s][l] / total_sums[k][s], 2);
                }
            }
            else {
                total_sums[k][s] = 0.0;
            }

            gini[k][s] += (1.0 - squared_fraction_all_labels) * total_sums[k][s];
        }
    }
}

pair<int, int> fhe_client::find_min_gini_feature_split_indices(const vector_2d<double>& gini) {
    pair<int, int> min_feature_split_indices = make_pair<int,int>(-1, -1);

    double min_gini = numeric_limits<double>::max();
    for (size_t k = 0; k < _features_count; k++) {
        for (size_t s = 0; s < _num_of_splits; s++) {
            if (gini[k][s] < min_gini) {
                min_feature_split_indices.first = k;
                min_feature_split_indices.second = s;
                min_gini = gini[k][s];
            }
        }
    }

    return  min_feature_split_indices;
}

vector_2d<Ciphertext> fhe_client::enc_one_hot_selector(size_t min_feature_index, size_t min_split_index)  {
    vector_2d<Ciphertext> one_hot_selector(_features_count, vector<Ciphertext>(_num_of_splits));

    for (size_t k = 0; k < _features_count; k++) {
        for (size_t s = 0; s < _num_of_splits; s++) {
            Plaintext ptxt;
            vector<double> ptxt_vector(_seal.slot_count, 0.0);

            if (k == min_feature_index && s == min_split_index) {
                ptxt_vector.assign(_seal.slot_count, 1.0);
            }

            _seal.encoder_ptr->encode(ptxt_vector, _seal.scale, ptxt);
            _seal.encryptor_ptr->encrypt(ptxt, one_hot_selector[k][s]);
        }
    }

    return one_hot_selector;
}

pair<int,int> fhe_client::gini_based_min_feature_split(vector_3d<double>& node_sums_left, vector_3d<double>& node_sums_right) {
    vector_2d<double> gini(_features_count,vector<double>(_num_of_splits, 0.0));

    smooth_sums(node_sums_left);
    add_single_side_to_gini(node_sums_left, gini);

    smooth_sums(node_sums_right);
    add_single_side_to_gini(node_sums_right, gini);

    return find_min_gini_feature_split_indices(gini);
}

tuple<vector_2d<Ciphertext>, Ciphertext, Ciphertext> fhe_client::generate_node_content(const vector_3d<Ciphertext> &node_sums_left, const vector_3d<Ciphertext> &node_sums_right) {
    vector_3d<double> plain_node_sums_left = dec_sums_by_feature_split_label(node_sums_left);
    vector_3d<double> plain_node_sums_right = dec_sums_by_feature_split_label(node_sums_right);
    vector<double> splits_doubles = encode_splits();

    pair<int,int> min_feature_split_indices = gini_based_min_feature_split(plain_node_sums_left, plain_node_sums_right);
    if (min_feature_split_indices.first == -1 || min_feature_split_indices.second == -1) {
        //ERROR
    }

    //Over Enc again!
    vector_2d<Ciphertext> one_hot_selector = enc_one_hot_selector(min_feature_split_indices.first, min_feature_split_indices.second);

    Plaintext plain_feature, plain_split;
    Ciphertext enc_feature, enc_split;

    _seal.encoder_ptr->encode(min_feature_split_indices.first, _seal.scale, plain_feature);
    _seal.encryptor_ptr->encrypt(plain_feature, enc_feature);
    _seal.encoder_ptr->encode(splits_doubles[min_feature_split_indices.second], _seal.scale, plain_split);
    _seal.encryptor_ptr->encrypt(plain_split, enc_split);

    tuple<vector_2d<Ciphertext>, Ciphertext, Ciphertext> res(one_hot_selector, enc_feature, enc_split);

    return res;
}

vector<shared_ptr<decision_tree_node>> fhe_client::decrypt_trained_tree(vector<shared_ptr<enc_tree_node>> trained_tree, size_t tree_height) const {
    size_t nodes_count = pow(2, tree_height + 1) - 1;
    size_t non_leaf_count = nodes_count - pow(2, tree_height);
    vector<shared_ptr<decision_tree_node>> nodes (nodes_count);

    // Handle non-leaves
    for (size_t i = 0; i < non_leaf_count; i++) {
        Plaintext plain_feature, plain_theta;
        vector<double> dec_feature, dec_theta;
        _seal.decryptor_ptr->decrypt(trained_tree[i]->get_feature_index(), plain_feature);
        _seal.encoder_ptr->decode(plain_feature, dec_feature);
        _seal.decryptor_ptr->decrypt(trained_tree[i]->get_theta(), plain_theta);
        _seal.encoder_ptr->decode(plain_theta, dec_theta);
        nodes[i] = make_shared<decision_tree_node>(i,dec_feature[0], dec_theta[0]);
    }

    // Handle leaves
    for (size_t i = non_leaf_count; i < nodes_count; i++) {
        vector<Ciphertext> leaf_enc_value = trained_tree[i]->get_leaf_value();
        vector<double> leaf_dec_value(leaf_enc_value.size(), 0.0);

        for (size_t j = 0; j < leaf_enc_value.size(); j++) {
            Plaintext plain_value;
            vector<double> value;
            _seal.decryptor_ptr->decrypt(leaf_enc_value[j], plain_value);
            _seal.encoder_ptr->decode(plain_value, value);
            for (size_t slot = 0; slot < _seal.slot_count; slot++) {
                leaf_dec_value[j] += value[slot];
            }
        }

        nodes[i] = make_shared<decision_tree_node>(i,leaf_dec_value);
    }

    // Set all nodes' children pointers
    for (size_t i = 0; i < non_leaf_count; i++) {
        nodes[i]->set_left_node(nodes[(2 * i) + 1]);
        nodes[i]->set_right_node(nodes[(2 * i) + 2]);
    }

    return nodes;
}

void fhe_client::save_plain_tree(const vector<shared_ptr<decision_tree_node>>& tree, const vector<size_t>& header, const string& destination) const {
    ofstream o_file (destination, ofstream::trunc);
    if (o_file.is_open())
    {
        o_file << header[0] << " " << header[1] << " " <<  header[2] << " ";

        for (size_t i = 0; i < tree.size(); i++) {
            if(tree[i]->is_leaf()) {
                for (int j = 0; j < _labels_count; j++) {
                    o_file << tree[i]->get_leaf_value()[j] << " ";
                }
            }
            else {
                o_file << tree[i]->get_feature_index() << " ";
                o_file << tree[i]->get_theta() << " ";
            }
        }
        o_file.close();
    } else throw invalid_argument("Unable to create tree files!");
}

vector<shared_ptr<decision_tree_node>> fhe_client::decrypt_and_save_tree(stringstream& source, const string& destination) const{
    string stream = source.str();
    vector<size_t> header(3); // (tree_height, features_count, labels_count)
    memcpy(&header[0], &stream.at(0), sizeof(header[0]));
    memcpy(&header[1], &stream.at(sizeof(header[0])), sizeof(header[1]));
    memcpy(&header[2], &stream.at(sizeof(header[1])), sizeof(header[2]));
    source.seekg(3 * sizeof(header[0]), source.beg);

    size_t nodes_count = (size_t)pow(2, header[0] + 1) - 1;
    size_t non_leaf_count = nodes_count - (size_t)pow(2, header[0]);
    vector<shared_ptr<enc_tree_node>> nodes (nodes_count);

    for (size_t i = 0; i < non_leaf_count; i++) {
        Ciphertext feature_index, theta;
        feature_index.load(*_seal.context_ptr, source);
        theta.load(*_seal.context_ptr, source);
        nodes[i] = make_shared<enc_tree_node>(i,feature_index,theta);
    }
    for (size_t i = non_leaf_count; i < nodes_count; i++) {
        vector<Ciphertext> values(_labels_count);
        for (int j = 0; j < _labels_count; j++) {
            values[j].load(*_seal.context_ptr, source);
        }
        nodes[i] = make_shared<enc_tree_node>(i,values);
    }
    for (size_t i = 0; i < non_leaf_count; i++) {
        nodes[i]->set_left_node(nodes[(2 * i) + 1]);
        nodes[i]->set_right_node(nodes[(2 * i) + 2]);
    }

    vector<shared_ptr<decision_tree_node>> dec_tree = decrypt_trained_tree(nodes, header[0]);
    if (!destination.empty()) save_plain_tree(dec_tree, header, destination);
    return dec_tree;
}

vector<double> fhe_client::decrypt_prediction(Ciphertext input) const {
    Plaintext plain_result;
    vector<double> result;
    _seal.decryptor_ptr->decrypt(input, plain_result);
    _seal.encoder_ptr->decode(plain_result, result);

    return result;
}

vector<Ciphertext> fhe_client::encrypt_instance(vector<double> input) const {
    vector<Plaintext> plain(_features_count);
    vector<Ciphertext> cipher(_features_count);
    for (size_t i = 0; i < input.size(); i++) {
        _seal.encoder_ptr->encode(input[i], _seal.scale, plain[i]);
        _seal.encryptor_ptr->encrypt(plain[i], cipher[i]);
    }

    return cipher;
}
