#ifndef FHE_RANDOM_FOREST_ENC_TREE_NODE_H
#define FHE_RANDOM_FOREST_ENC_TREE_NODE_H

#include "utils.h"
#include <fstream>
#include <utility>
#include <unistd.h>

/** 
 *
 * @Description:
 * A node of an encrypted decision tree.
 *
 * @note
 * 3 types of constructors:
 * 1. basic node, receives index, feature and theta.
 * 2. full node, receives index, feature, theta and child nodes.
 * 3. leaf node, receives index and leaf value.
 *
 */
class enc_tree_node {
private:
    int _index;
    shared_ptr<enc_tree_node> _left, _right;
    bool _is_leaf;

    // leaf fields
    vector<Ciphertext> _leaf_value;

    // non-leaf fields
    Ciphertext _feature_index;
    Ciphertext _theta;

public:
    // basic node
    enc_tree_node(int index, Ciphertext feature_index, Ciphertext theta) :
            _index(index),
            _is_leaf(false),
            _feature_index(feature_index),
            _theta(theta),
            _leaf_value(0){}

    // full node
    enc_tree_node(int index, Ciphertext feature_index, Ciphertext theta, std::shared_ptr<enc_tree_node> left, std::shared_ptr<enc_tree_node> right) :
            _index(index),
            _is_leaf(false),
            _feature_index(feature_index),
            _theta(theta),
            _left(std::move(left)),
            _right(std::move(right)),
            _leaf_value(0) {}

    // leaf node
    enc_tree_node(int index, vector<Ciphertext>& leaf_value) :
            _index(index),
            _is_leaf(true),
            _leaf_value(leaf_value) {}

    enc_tree_node(const enc_tree_node& other) = default;
    enc_tree_node(enc_tree_node&& other)  = default;
    enc_tree_node& operator=(const enc_tree_node& other) = default;
    enc_tree_node& operator=(enc_tree_node&& other) = default;
    virtual ~enc_tree_node() = default;

    Ciphertext get_feature_index() const {
        return _feature_index;
    }

    Ciphertext get_theta() const {
        return _theta;
    }

    vector<Ciphertext> get_leaf_value() const {
        return _leaf_value;
    }

    int get_index() const{
        return _index;
    }

    bool is_leaf() const{
        return _is_leaf;
    }

    void set_left_node(shared_ptr<enc_tree_node>& node) {
        _left = node;
    }

    void set_right_node(shared_ptr<enc_tree_node>& node) {
        _right = node;
    }

    bool has_left_node() {
        return static_cast<bool>(_left);
    }

    bool has_right_node() {
        return static_cast<bool>(_right);
    }

    shared_ptr<enc_tree_node> get_left_node() {
        return _left;
    }

    shared_ptr<enc_tree_node> get_right_node() {
        return _right;
    }

};


#endif //FHE_RANDOM_FOREST_ENC_TREE_NODE_H
