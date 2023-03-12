#ifndef FHE_RANDOM_FOREST_DECISION_TREE_NODE_H
#define FHE_RANDOM_FOREST_DECISION_TREE_NODE_H

#include "utils.h"

using namespace std;

/** 
 *
 * @Description:
 * A node of a plain decision tree.
 *
 * @note
 * 3 types of constructors:
 * 1. basic node, receives index, feature and theta.
 * 2. full node, receives index, feature, theta and child nodes.
 * 3. leaf node, receives index and leaf value.
 *
 */
class decision_tree_node {
private:
    int _index;
    shared_ptr<decision_tree_node> _left, _right;
    bool _is_leaf;

    // leaf fields
    vector<double> _leaf_value;

    // non-leaf fields
    int _feature_index;
    double _theta;

    /**
     * Print the tree starting at this node
     *
     * @param prefix Prefix for printing.
     * @param is_right Bool declaring if current node is the right node, or left.
     *
     * @note
     * Used by the public print() function of the node.
     *
     */
    void print(const string& prefix, bool is_right) {
        cout << prefix;
        cout << (is_right ? "├──" : "└──" );

        if (!_is_leaf) {
            cout << "{" << _index << "} : [" << _feature_index  << ", " << _theta << "]" << std::endl;
        }
        else {
            cout << "{" << _index << "} : (";
            for (int i = 0; i<size(_leaf_value); i++) {
                cout <<  _leaf_value[i] << (i==size(_leaf_value)-1?"":" ");
            }
            cout << ")" << endl;
        }

        if (_right) {
            _right->print(prefix + (is_right ? "│   " : "    "), true);
        }

        if (_left) {
            _left->print(prefix + (is_right ? "│   " : "    "), false);
        }

    }

public:
    // basic node
    decision_tree_node(int index, int feature_index, double theta) :
            _index(index),
            _is_leaf(false),
            _feature_index(feature_index),
            _theta(theta),
            _leaf_value(0){}

    // full node
    decision_tree_node(int index, uint feature_index, double theta, std::shared_ptr<decision_tree_node> left, std::shared_ptr<decision_tree_node> right) :
            _index(index),
            _is_leaf(false),
            _feature_index(feature_index),
            _theta(theta),
            _left(std::move(left)),
            _right(std::move(right)),
            _leaf_value(0) {}

    // leaf node
    decision_tree_node(int index, vector<double>& leaf_value) :
            _index(index),
            _is_leaf(true),
            _leaf_value(leaf_value),
            _feature_index(0),
            _theta(0) {}

    decision_tree_node(const decision_tree_node& other) = default;
    decision_tree_node(decision_tree_node&& other)  = default;
    decision_tree_node& operator=(const decision_tree_node& other) = default;
    decision_tree_node& operator=(decision_tree_node&& other) = default;
    virtual ~decision_tree_node() = default;

    int get_feature_index() const {
        return _feature_index;
    }

    double get_theta() const {
        return _theta;
    }

    vector<double> get_leaf_value() const {
        return _leaf_value;
    }

    int get_index() const{
        return _index;
    }

    bool is_leaf() const{
        return _is_leaf;
    }

    void set_left_node(shared_ptr<decision_tree_node>& node) {
        _left = node;
    }

    void set_right_node(shared_ptr<decision_tree_node>& node) {
        _right = node;
    }

    bool has_left_node() {
        return static_cast<bool>(_left);
    }

    bool has_right_node() {
        return static_cast<bool>(_right);
    }

    shared_ptr<decision_tree_node> get_left_node() {
        return _left;
    }

    shared_ptr<decision_tree_node> get_right_node() {
        return _right;
    }

    /**
     * Print the tree starting at this node
     */
    void print() {
        print("", false);
    }

};


#endif //FHE_RANDOM_FOREST_DECISION_TREE_NODE_H
