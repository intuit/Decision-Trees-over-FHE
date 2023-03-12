#ifndef FHE_RANDOM_FOREST_DATA_TYPES_H
#define FHE_RANDOM_FOREST_DATA_TYPES_H

#include <vector>
#include <memory>
#include "seal/seal.h"

using namespace std;
using namespace seal;

using PublicKeys = pair<PublicKey, RelinKeys>;

/** 
 *
 * @b Description:
 * Extended SEAL context: contains the SEAL parameters, an initialized SEAL CKKS context,
 * and other objects related to the initialized context 
 *
 * @note
 * A SEAL_c object has 2 modes:
 * - Encryption only. Public key is set by set_public_keys()
 * - Encryption and decryption. Keys are set by generate_keys()
 * (No option to set the secret key from outside. Once set_public_keys() is called, the object is in encryption only mode)
 *
 */

class SEAL_c {
public:
    shared_ptr<SEALContext> context_ptr;
    shared_ptr<KeyGenerator> keygen_ptr;

    shared_ptr<Encryptor> encryptor_ptr;
    shared_ptr<Evaluator> evaluator_ptr;
    shared_ptr<Decryptor> decryptor_ptr;

    shared_ptr<CKKSEncoder> encoder_ptr;
    shared_ptr<PublicKey> public_key;
    shared_ptr<RelinKeys> relin_keys;

    double scale;
    size_t slot_count;

    SEAL_c() {
        scale = 0;
        slot_count = 0;
        context_ptr = nullptr;
        keygen_ptr = nullptr;
        encryptor_ptr = nullptr;
        evaluator_ptr = nullptr;
        decryptor_ptr = nullptr;
        encoder_ptr = nullptr;
        relin_keys = nullptr;
    }

    SEAL_c(int sealScale, int polyModulusDegree, vector<int> bitSizes) {
        EncryptionParameters parms(scheme_type::ckks);

        scale = pow(2.0, sealScale);
        size_t poly_modulus_degree = pow(2, polyModulusDegree);
        vector<int> bit_sizes = std::move(bitSizes);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, bit_sizes));
        context_ptr = make_shared<SEALContext>(parms, true, sec_level_type::tc128);
        keygen_ptr = make_shared<KeyGenerator>(*context_ptr);
        evaluator_ptr = make_shared<Evaluator>(*context_ptr);
        encoder_ptr = make_shared<CKKSEncoder>(*context_ptr);
        slot_count = encoder_ptr->slot_count();
    }

    PublicKeys generate_keys() {
        PublicKeys keys;
        keygen_ptr->create_public_key(keys.first);
        public_key = make_shared<PublicKey>(keys.first);
        encryptor_ptr = make_shared<Encryptor>(*context_ptr, keys.first);
        keygen_ptr->create_relin_keys(keys.second);
        relin_keys = make_shared<RelinKeys>(keys.second);
        decryptor_ptr = make_shared<Decryptor>(*context_ptr, keygen_ptr->secret_key());

        return keys;
    }

    void set_public_keys(const PublicKeys& publicKeys) {
        encryptor_ptr = make_shared<Encryptor>(*context_ptr, publicKeys.first);
        public_key = make_shared<PublicKey>(publicKeys.first);
        relin_keys = make_shared<RelinKeys>(publicKeys.second);
    }

    // In place alignment of two input ciphertexts to the lower modulus size
    inline void align_modulus(Ciphertext& left, Ciphertext& right) const {
        if (evaluator_ptr) {
            if (left.coeff_modulus_size() > right.coeff_modulus_size()) {
                evaluator_ptr->mod_switch_to_inplace(left, right.parms_id());
            } else if (left.coeff_modulus_size() < right.coeff_modulus_size()) {
                evaluator_ptr->mod_switch_to_inplace(right, left.parms_id());
            }
        }
    }

    // Out of place alignment of two input ciphertexts to the lower modulus size
    inline const Ciphertext& align_modulus_to(const Ciphertext& left, const Ciphertext& right, Ciphertext& destination) {
        if (evaluator_ptr) {
            if (left.coeff_modulus_size() > right.coeff_modulus_size()) {
                evaluator_ptr->mod_switch_to(left, right.parms_id(),destination);
                return right;
            } else {
                evaluator_ptr->mod_switch_to(right, left.parms_id(), destination);
                return left;
            }
        }
        return left;
    }
};

template<class T>
using Sample = pair<vector<T>, vector<T>>;

template<class T>
vector<T>& access_features(Sample<T>& xy) { return xy.first; }

template<class T>
vector<T>& access_labels(Sample<T>& xy) { return xy.second; }

template<class T>
void sample_init(Sample<T>& xy, size_t features_count, size_t labels_count) {
    xy.first.resize(features_count);
    xy.second.resize(labels_count);
}

template<class T>
using vector_2d = vector<vector<T>>;

template<class T>
using vector_3d = vector<vector<vector<T>>>;

template<class T>
using vector_4d = vector<vector<vector<vector<T>>>>;

template<class T>
using TrainingData = tuple<vector_3d<T>, vector_3d<T>, vector_4d<T>, vector_4d<T>>;

template<class T>
vector_3d<T>& access_soft_ifs_right(TrainingData<T>& training_data) {
    return get<0>(training_data);
}

template<class T>
vector_3d<T>& access_soft_ifs_left(TrainingData<T>& training_data) {
    return get<1>(training_data);
}

template<class T>
vector_4d<T>& access_labeled_soft_ifs_right(TrainingData<T>& training_data) {
    return get<2>(training_data);
}

template<class T>
vector_4d<T>& access_labeled_soft_ifs_left(TrainingData<T>& training_data) {
    return get<3>(training_data);
}

// Initialize the vector sizes of the TrainingData 4-dimensions matrix.
template<class T>
void training_data_init(TrainingData<T>& training_data, size_t chunks, size_t features, size_t splits, size_t labels) {
    // chunks * features * splits
    vector_3d<T>& soft_ifs_right = access_soft_ifs_right<T>(training_data);
    vector_3d<T>& soft_ifs_left = access_soft_ifs_left<T>(training_data);

    soft_ifs_right.resize(chunks);
    soft_ifs_left.resize(chunks);
    for (size_t i = 0; i < chunks; i++) {
        soft_ifs_right[i].resize(features);
        soft_ifs_left[i].resize(features);
        for (size_t k = 0; k < features; k++) {
            soft_ifs_right[i][k].resize(splits);
            soft_ifs_left[i][k].resize(splits);
        }
    }

    // chunks * features * splits * labels
    vector_4d<T>& labeled_soft_ifs_right = access_labeled_soft_ifs_right<T>(training_data);
    vector_4d<T>& labeled_soft_ifs_left = access_labeled_soft_ifs_left<T>(training_data);

    labeled_soft_ifs_right.resize(chunks);
    labeled_soft_ifs_left.resize(chunks);

    for (size_t i = 0; i < chunks; i++) {
        labeled_soft_ifs_right[i].resize(features);
        labeled_soft_ifs_left[i].resize(features);
        for (size_t k = 0; k < features; k++) {
            labeled_soft_ifs_right[i][k].resize(splits);
            labeled_soft_ifs_left[i][k].resize(splits);

            for (size_t s = 0; s < splits; s++) {
                labeled_soft_ifs_right[i][k][s].resize(labels);
                labeled_soft_ifs_left[i][k][s].resize(labels);
            }
        }
    }
}

#endif //FHE_RANDOM_FOREST_DATA_TYPES_H
