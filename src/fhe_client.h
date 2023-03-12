#ifndef FHE_RANDOM_FOREST_FHE_CLIENT_H
#define FHE_RANDOM_FOREST_FHE_CLIENT_H

#include <fstream>
#include <utility>
#include "data_types.h"
#include "utils.h"
#include "soft_if.h"
#include "decision_tree_node.h"
#include "enc_tree_node.h"
#include <gtest/gtest.h>


/** Class that represents the client and it's functionality.
 *
 * @b Description:
 * The client generates a key pair, encrypts the data, and sends it (together with the public key) to the server.
 * The server trains and returns to the client an encrypted tree.
 *
 * @note
 * The client also has a public function called generate_node_content, which the server uses in order to compute gini impurity in cleartext.
 *
 */
class fhe_client {
private:
    SEAL_c _seal;
    PublicKeys _public_keys;
    size_t _features_count = 0, _labels_count = 0;
    double _split_initial_value;
    double _split_final_value;
    size_t _num_of_splits;
    double _split_step_size;
    vector<Sample<Ciphertext>> _enc_data;
    size_t _chunks = 0;
    size_t _samples = 0;

    // For testing purposes.
    FRIEND_TEST(GiniTest, PyCompare);

    /**
    * Generates splits vector with encoded thresholds.
    */
    [[nodiscard]] vector<double> encode_splits() const;

    /**
     * Encrypts input plain data.
     *
     * @param plain_data The samples for training, in plaintext
     * @param chunks number of chunks to divide the data to.
     *
     * @retval The encrypted data.
     *
     * @note
     * The data is divided into chunks in order to take advantage of the batching capability of CKKS.
     */
    vector<Sample<Ciphertext>> encrypt_packed_samples(vector<Sample<double>>& plain_data, size_t chunks);

    /**
     * Decrypts weighted sums per label for each potential threshold and feature.
     *
     * @param enc_sums The encrypted 3d vector.
     *
     * @retval The decrypted vector.
     */
    vector_3d<double> dec_sums_by_feature_split_label(const vector_3d<Ciphertext>& enc_sums);

    /**
     * Smooths in-place the 3d vector values; zeroes negative sums and rounds all sums.
     *
     * @param sums The 3d vector to be smoothed.
     */
    void smooth_sums(vector_3d<double>& sums);

    /**
     * Calculates the Gini impurity of all features and thresholds of a single side (left or right).
     * THe result is added to an input gini vector.
     *
     * @param sums The single side (left or right) sums.
     * @param gini the gini vector to add the result to.
     */
    void add_single_side_to_gini(const vector_3d<double>& sums, vector_2d<double>& gini);

    /**
     * Finds the feature and threshold pair that minimizes the gini impurity.
     *
     * @param gini The gini scores for all possible (feature, threshold) pairs.
     *
     * @retval The indices of the minimal feature and threshold, as a pair: (feature index, threshold index)
     */
    pair<int, int> find_min_gini_feature_split_indices(const vector_2d<double>& gini);

    /**
     * Generates and encrypts a one-hot-encoding of the selected feature and threshold.
     *
     * @param min_feature_index selected feature index.
     * @param min_split_index selected threshold index.
     *
     * @retval A 2d vector of ciphertexts that encrypt the one-hot-encodings for (feature, threshold).
     */
    vector_2d<Ciphertext> enc_one_hot_selector(size_t min_feature_index, size_t min_split_index);

    /**
     * Finds the indices of the feature and threshold that minimize the gini impurity
     *
     * @param node_sums_left  left side weighted sums per label for each potential threshold and feature.
     * @param node_sums_right right side weighted sums per label for each potential threshold and feature.
     *
     * @retval The indices of the found feature and threshold, as a pair: (feature, threshold)
     *
     */
    pair<int,int> gini_based_min_feature_split(vector_3d<double>& node_sums_left, vector_3d<double>& node_sums_right);

    /**
     * Decrypts a trained tree (trained by tree_train_server).
     *
     * @param trained_tree The encrypted tree (as a vector of pointers to nodes, root at 0).
     * @param tree_height The tree height.
     *
     * @retval The decrypted tree (as a vector of pointers to nodes, root at 0).
     *
     * @see
     * tree_train_server.execute()
     */
    [[nodiscard]] vector<shared_ptr<decision_tree_node>> decrypt_trained_tree(vector<shared_ptr<enc_tree_node>> trained_tree, size_t tree_height) const;

    /**
     * Saving a serialized plain decision tree on file, to be used by the eval server.
     *
     * @param tree The plain tree to be saved.
     * @param header Headers to be added to the serialization (tree height, labels count and features count).
     * @param destination The path where to save the file.
     */
    void save_plain_tree(const vector<shared_ptr<decision_tree_node>>& tree, const vector<size_t>& header, const string& destination) const;

public:
    fhe_client(int scale, int polyModulusDegree, vector<int> bitSizes, double split_step_size = 0.05,
    double split_initial_value = -1.0, double split_final_value = 1.0) {
        _seal = {scale, polyModulusDegree, bitSizes};
        _public_keys = _seal.generate_keys();
        _split_initial_value = split_initial_value;
        _split_final_value = split_final_value;
        _split_step_size = split_step_size;
        _num_of_splits = ceil((_split_final_value - _split_initial_value) / _split_step_size) + 1;
    }

    PublicKeys get_public_keys() const { return _public_keys; }

    /**
     * Computes the values (feature and threshold) of a node, by computing the gini impurity in cleartext
     *
     * @param node_sums_left The encrypted sums of weighs the left side of the node.
     * @param node_sums_right The encrypted sums of weighs of the right side of the node.
     *
     * @retval The encrypted feature and threshold values selected for the node,
     * and the encrypted one-hot-encoding of their indices for updating the weights passed on to the node's children
     *
     * @note
     * This function is called by the server.
     */
    tuple<vector_2d<Ciphertext>, Ciphertext, Ciphertext> generate_node_content(const vector_3d<Ciphertext> &node_sums_left, const vector_3d<Ciphertext> &node_sums_right);

    /**
     * Reads the input data samples from file and encrypts them.
     *
     * @param plain_file_full_path The full path to the data file.
     * @param features_count number of features.
     * @param labels_count number of labels.
     *
     * @retval The encrypted data.
     */
    vector<Sample<Ciphertext>> get_encrypted_data(string& plain_file_full_path, size_t features_count, size_t labels_count);

    /**
     * Deserializing a plain decision tree from datastream retrieved from the training server, and saving the decrypted
     * decision tree serialized on file.
     *
     * @param source The data stream recieved from the training server.
     * @param destination The path where to save the serialized decrypted decision tree.
     *
     * @retval The plain decision tree.
     */
    vector<shared_ptr<decision_tree_node>> decrypt_and_save_tree(stringstream& source, const string& destination) const;

    /**
     * Encrypting an instance of values to be predicted on the eval server.
     *
     * @param input the instance to be encrypted.
     *
     * @retval The encrypted instance, as a Ciphertext vector.
     */
    vector<Ciphertext> encrypt_instance(vector<double> input) const;

    /**
     * Decrypting the prediction result, given as a vector of Ciphertexts generated by the eval server.
     *
     * @param input The result generated by the eval server.
     *
     * @retval The encrypted result, as a vector of doubles.
     */
    vector<double> decrypt_prediction(Ciphertext input) const;
};

#endif //FHE_RANDOM_FOREST_FHE_CLIENT_H
