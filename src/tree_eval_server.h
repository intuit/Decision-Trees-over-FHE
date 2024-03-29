#ifndef FHE_RANDOM_FOREST_EVALUATION_H
#define FHE_RANDOM_FOREST_EVALUATION_H

#include "utils.h"
#include "decision_tree_node.h"
#include "soft_if.h"
#include <gtest/gtest.h>

using namespace std;
using namespace seal;

/** 
 *
 * @b Description:
 * Class that predicts the label of a given encrypted sample, using a plain decision tree.
 *
 */
class tree_eval_server {
private:
    SEAL_c _seal;
    soft_if _soft_if_op = {_seal, 0};

    // For testing purposes.
    FRIEND_TEST(TreeEval, differentPolyDegrees);

    /**
     * Predicts the label of a given encrypted sample (vector of features), using a plain decision tree.
     *
     * @param node The root of the decision tree to use.
     * @param x The encrypted sample to predict the label of.
     *
     * @retval The prediction, as a Ciphertext.
     */
    Ciphertext predict(decision_tree_node& node, vector<Ciphertext>& x);

    /**
     * Deserializing a plain decision tree from file generated by the client.
     *
     * @param source The path to the decision tree to deserialize.
     *
     * @retval The plain decision tree.
     */
    [[nodiscard]] static vector<shared_ptr<decision_tree_node>> load_plain_tree(const string& source);

public:
    tree_eval_server(int sealScale, int polyModulusDegree, vector<int> bitSizes, int polyDegree, const PublicKeys& keys) {
        _seal = {sealScale, polyModulusDegree, bitSizes};
        _seal.set_public_keys(keys);
        _soft_if_op = {_seal, polyDegree};
    }

    tree_eval_server(SEAL_c seal, int polyDegree) {
        _seal = std::move(seal);
        _soft_if_op = {_seal, polyDegree};
    }

    /**
     * Predicts the label of a given encrypted sample (vector of features), by deserializing a plain tree from file.
     *
     * @param source The path to the decision tree to use.
     * @param x The encrypted sample to predict the label of.
     *
     * @retval The prediction, as a Ciphertext.
     */
    Ciphertext execute(const string& source, std::vector<Ciphertext>& x);
};


#endif //FHE_RANDOM_FOREST_EVALUATION_H
