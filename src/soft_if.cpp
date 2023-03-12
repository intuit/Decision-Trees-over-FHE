#include "soft_if.h"

void soft_if::soft_if_8_calc_powers(const int max_2_power, vector<Ciphertext>& cipher_pows, const Ciphertext& x) const{
    cipher_pows[0] = x;

    //Compute encrypted powers of 2 into cipher_pows vector
    for (size_t i = 2; i <= max_2_power; i *= 2) {
        _seal.evaluator_ptr->square(cipher_pows[(i/2)-1], cipher_pows[i-1]);
        _seal.evaluator_ptr->relinearize_inplace(cipher_pows[i-1], *_seal.relin_keys);
        _seal.evaluator_ptr->rescale_to_next_inplace(cipher_pows[i-1]);
    }

    //  x^3 = x * x^2
    _seal.evaluator_ptr->mod_switch_to_inplace(cipher_pows[0], cipher_pows[1].parms_id());
    _seal.evaluator_ptr->multiply(cipher_pows[0], cipher_pows[1], cipher_pows[2]);
    _seal.evaluator_ptr->relinearize_inplace(cipher_pows[2], *_seal.relin_keys);
    _seal.evaluator_ptr->rescale_to_next_inplace(cipher_pows[2]);

    //  x^5 = x^2 * x^3
    _seal.evaluator_ptr->mod_switch_to_inplace(cipher_pows[1], cipher_pows[2].parms_id());
    _seal.evaluator_ptr->multiply(cipher_pows[1], cipher_pows[2], cipher_pows[4]);
    _seal.evaluator_ptr->relinearize_inplace(cipher_pows[4], *_seal.relin_keys);
    _seal.evaluator_ptr->rescale_to_next_inplace(cipher_pows[4]);

    //  x^7 = x^3 * x^4
    _seal.evaluator_ptr->multiply(cipher_pows[2], cipher_pows[3], cipher_pows[6]);
    _seal.evaluator_ptr->relinearize_inplace(cipher_pows[6], *_seal.relin_keys);
    _seal.evaluator_ptr->rescale_to_next_inplace(cipher_pows[6]);
}

void soft_if::soft_if_sparse_internal(const vector<double>& coeffs, vector<Ciphertext> cipher_pows, Ciphertext& cipher_result, const Ciphertext& x) const{
    if (coeffs.size()%8 != 0){ // in fact size must be a power of 2 (and >= 8)
        throw invalid_argument( "unexpected number of coefficents" );
    }

    //Recursively dividing to soft ifs of poly degree 8
    if (coeffs.size() == 8){
        const int polynomial_degree = 7;

        //Encode polynomial coefficients into 'plain_coeffs' - a vector of Plaintext objects
        //and mult coeffs of x^1 to x^8
        vector<Plaintext> plain_coeffs (polynomial_degree);
        for (int i = 0; i < polynomial_degree; i++) {
            if (coeffs[i] != 0.0) {
                _seal.encoder_ptr->encode(coeffs[i], cipher_pows[i].scale(), plain_coeffs[i]);
                _seal.evaluator_ptr->mod_switch_to_inplace(plain_coeffs[i], cipher_pows[i].parms_id());
                _seal.evaluator_ptr->multiply_plain_inplace(cipher_pows[i], plain_coeffs[i]);
                _seal.evaluator_ptr->rescale_to_next_inplace(cipher_pows[i]);
            }
        }

        //Align scale for all ciphers
        //Modulus switch to the lowest modulus
        parms_id_type last_parms_id = cipher_pows[polynomial_degree - 1].parms_id();
        for (int i = 0; i < polynomial_degree - 1; i++) {
            if (coeffs[i] != 0.0){
                cipher_pows[i].scale() = cipher_pows[polynomial_degree - 1].scale();
                _seal.evaluator_ptr->mod_switch_to_inplace(cipher_pows[i], last_parms_id);
            }
        }

        //Add all monomials
        cipher_result = cipher_pows[0];
        _seal.evaluator_ptr->add_inplace(cipher_result, cipher_pows[2]);
        _seal.evaluator_ptr->add_inplace(cipher_result, cipher_pows[4]);
        _seal.evaluator_ptr->add_inplace(cipher_result, cipher_pows[6]);
    }
    else{
        Ciphertext cipher_result_1;

        soft_if_sparse_internal(vector<double>(coeffs.begin(), coeffs.begin() + coeffs.size()/2), cipher_pows, cipher_result_1, x);
        soft_if_sparse_internal(vector<double>(coeffs.begin() + coeffs.size()/2, coeffs.end()), cipher_pows, cipher_result, x);
        _seal.evaluator_ptr->mod_switch_to_inplace(cipher_pows[coeffs.size()/2-1], cipher_result.parms_id());
        _seal.evaluator_ptr->multiply_inplace(cipher_result, cipher_pows[coeffs.size()/2-1]);
        _seal.evaluator_ptr->relinearize_inplace(cipher_result, *_seal.relin_keys);
        _seal.evaluator_ptr->rescale_to_next_inplace(cipher_result);

        cipher_result_1.scale() = cipher_result.scale();
        _seal.align_modulus(cipher_result, cipher_result_1);
        //_seal.evaluator_ptr->mod_switch_to_inplace(cipher_result_1, cipher_result.parms_id());
        _seal.evaluator_ptr->add_inplace(cipher_result, cipher_result_1);
        //_seal.evaluator_ptr->rescale_to_next_inplace(cipher_result);
    }
}

void soft_if::apply(Ciphertext& destination, const Ciphertext& source) const {
    vector<Ciphertext> cipher_pows(max(_coeffs.size()-1, _coeffs.size()/2));
    soft_if_8_calc_powers(_coeffs.size()/2, cipher_pows, source);
    soft_if_sparse_internal(_coeffs, cipher_pows, destination, source);

    //Add the free coefficient which wasn't added in soft_if_sparse_internal
    Plaintext plain_free_coeff;
    _seal.encoder_ptr->encode(_free_coeff, destination.scale(), plain_free_coeff);
    _seal.evaluator_ptr->mod_switch_to_inplace(plain_free_coeff, destination.parms_id());
    _seal.evaluator_ptr->add_plain_inplace(destination, plain_free_coeff);
}

void soft_if::operator()(Ciphertext& destination, const Ciphertext& source) const {
    return apply(destination, source);
}
