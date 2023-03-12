#ifndef FHE_RANDOM_FOREST_TREE_TRAIN_SERVER_H
#define FHE_RANDOM_FOREST_TREE_TRAIN_SERVER_H

#include <queue>
#include "data_types.h"
#include "utils.h"
#include "soft_if.h"
#include "fhe_client.h"
#include "enc_tree_node.h"

/** Class that represents the tree training server
 *
 * @b Description:
 * The server trains a decision tree over encrypted data provided by the client, along with encryption context parameters
 * needed for the computation
 *
 * @note
 * All server functions operate over encrypted data without any access to the secret key.
 * The server uses the client's public function generate_node_content.
 *
 */
class tree_train_server {
private:
    SEAL_c _seal;
    fhe_client& _client;
    size_t _features_count, _labels_count, _soft_if_degree;
    vector<Plaintext> _splits;
    vector<Sample<Ciphertext>> _data;
    size_t _tree_height;

    /**
     * Process the training data for training optimization purposes.
     * For each feature and threshold combination, compute:
     *              Right Side:
     *                    1. soft_if(feature_value - threshold)
     *                    2. soft_if(feature_value - threshold) multiplied by one-hot-encoding of the label
     *              Left Side:
     *                    1. soft_if(threshold - feature_value)
     *                    2. soft_if(threshold - feature_value) multiplied by one-hot-encoding of the label
     *
     * @retval The results of the above computations
     */
    TrainingData<Ciphertext> prepare_training_data();

    /**
     * Generate a weighted sum, per label, of the training data for each (feature, threshold) combination
     *
     * @param weights The weights.
     * @param labeled_soft_ifs The calculated soft-if for each feature value (from training data) and threshold. Represented as a vector,
     * where the value is located in the entry corresponding to the label
     *
     * @retval The result sums per feature, threshold and label.
     *.
     */
    vector_3d<Ciphertext> weighted_sum_by_feature_split_label(const vector<Ciphertext>& weights, const vector_4d<Ciphertext>& labeled_soft_ifs);

    /**
     * Generate weights based on parent weights (weights of parent node) and selected feature and threshold for current node, as encoded by one hot-- selector.
     *
     * @param soft_ifs pre-calculated soft-if values for each (feature, threshold) combination, for each data sample.
     * @param one_hot_selector one hot encoding of selected feature and threshold.
     * @param parent_weights The weights of the parent node.
     * @param chunks The number of ciphertexts.
     * @param is_root
     *
     * @retval The updated weights.
     *
     */
    vector<Ciphertext> update_weights(const vector_3d<Ciphertext>& soft_ifs, const vector_2d<Ciphertext>& one_hot_selector,
                                      const vector<Ciphertext>& parent_weights, size_t chunks, bool is_root = false);

    /**
     * Generate the content of a leaf in the trained decision tree: vector of values per label
     *
     * @param weights_by_node_queue BFS queue containing weights of not-yet-processed nodes.
     *
     * @retval A vector of leaf values per label
     */
    vector<Ciphertext> process_leaf(queue<vector<Ciphertext>>& weights_by_node_queue);

    /**
     * Generates the content of an internal node in the training tree: (feature, threshold) pair. Also calculates the weights for the children
     * nodes and pushes them to the BFS queue
     *
     * @param weights_by_node_queue BFS queue containing weights of not-yet-processed nodes.
     * @param training_data THe processed training data (output of prepare_training_data)
     * @param chunks The number of ciphertexts.
     * @param index The node's index (0 if root)
     *
     * @retval The node values as a (feature, threshold) pair.
     *
     */
    pair<Ciphertext, Ciphertext> process_node(queue<vector<Ciphertext>>& weights_by_node_queue, TrainingData<Ciphertext>& training_data, size_t chunks, size_t index);

    /**
     * Streaming a serialized encrypted decision tree on a data-stream, to be read by the client.
     *
     * @param tree The encrypted tree to be streamed.
     * @param destination The data-stream to use.
     */
    void stream_tree(const vector<shared_ptr<enc_tree_node>>& tree, stringstream& destination) const;

public:
    tree_train_server(int scale, int polyModulusDegree, vector<int> bitSizes, const PublicKeys public_keys, fhe_client& client, size_t features_count,
                      size_t labels_count, size_t tree_height, size_t soft_if_degree, vector<Sample<Ciphertext>> data, double split_step_size = 0.05,
                      double split_initial_value = -1.0, double split_final_value = 1.0) :
            _client(client), _features_count(features_count), _labels_count(labels_count), _tree_height(tree_height),
            _soft_if_degree(soft_if_degree), _data(data) {

        _seal = {scale, polyModulusDegree, std::move(bitSizes)};
        _seal.set_public_keys(public_keys);

        size_t splits_count = (size_t)ceil((split_final_value - split_initial_value) / split_step_size) + 1;
        _splits.resize(splits_count);
        double current_value = split_initial_value;
        for (size_t s = 0; s < splits_count; s++) {
            _seal.encoder_ptr->encode(current_value, _seal.scale, _splits[s]);
            current_value += split_step_size;
        }
    }

    /**
     * Train an encrypted tree with given height
     *
     * @param tree_height The desired tree height.
     *
     * @retval The trained encrypted tree
     */
    void execute(stringstream& datastream);
};


#endif //FHE_RANDOM_FOREST_TREE_TRAIN_SERVER_H
