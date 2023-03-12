#ifndef FHE_RANDOM_FOREST_SOFT_IF_H
#define FHE_RANDOM_FOREST_SOFT_IF_H

#include "data_types.h"

/**
 * Pre-determined coefficients for the soft-if polynomial approximating the step function, with degrees 8,16,32.
 */
#define COEFFS8 {\
0.98197868, 0, -0.59230294, 0,\
0.16746853, 0, -0.01605682, 0\
}
#define FREE_COEFF8 5.00000000e-01

#define COEFFS16 {\
1.17176558e+00, 0, -1.26931700e+00, 0,\
8.77246161e-01, 0, -3.45040130e-01, 0,\
7.79988023e-02, 0, -9.86764478e-03, 0,\
6.30289488e-04, 0,-1.48856896e-05, 0\
}
#define FREE_COEFF16 5.00000000e-01

#define COEFFS32 {\
2.45383240e+00, 0, -1.19512267e+01, 0,\
3.92838952e+01, 0, -8.04307117e+01, 0,\
1.08618110e+02, 0, -1.01475361e+02, 0,\
6.78289165e+01, 0, -3.31693868e+01, 0,\
1.20214168e+01, 0, -3.24385239e+00, 0,\
6.49089058e-01, 0, -9.49345593e-02, 0,\
9.85629742e-03, 0, -6.87727000e-04, 0,\
2.89102547e-05, 0, -5.53185100e-07, 0\
}
#define FREE_COEFF32 4.99999992e-01

/** Class that calculates a soft-if polynomial.
 *
 * @b Description:
 * Given a Ciphertext x, evaluates the soft-if polynomial.
 *
 * @note
 * Constructor receives  a seal pointer (to calculate with) and a polynomial degree, which determines which pre-determined
 * soft-if coefficients to use in the function.
 *
 */
class soft_if {
protected:
    SEAL_c _seal;
    int _poly_degree;
    vector<double> _coeffs;
    double _free_coeff;

    /**
     * Apply the soft if function to a given source.
     *
     * @param source The Ciphertext to apply the function on.
     * @param destination The Ciphertext to put the result in.
     *
     */
    void apply(Ciphertext& destination, const Ciphertext& source) const;

    /**
     * Pre-calculate x powers for soft if 8.0
     *
     * @param x The Ciphertext to calculate the powers of.
     * @param cipher_pows The Ciphertexts vector to put the results in.
     * @param max_2_power Maximum power of 2 to pre-calculate.
     *
     */
    void soft_if_8_calc_powers(const int max_2_power, vector<Ciphertext>& cipher_pows, const Ciphertext& x) const;

    /**
     * Calculate the soft if Ciphertext result using pre-calculated powers.
     *
     * @param x The Ciphertext to be applied to the function.
     * @param cipher_result The Ciphertext to put the result in.
     * @param cipher_pows The pre-calculated powers of x.
     * @param coeffs The coefficients to use in the current iteration.
     *
     * @note
     * The free coefficient needs to be added afterwards!
     */
    void soft_if_sparse_internal(const vector<double>& coeffs, vector<Ciphertext> cipher_pows, Ciphertext& cipher_result, const Ciphertext& x) const;

public:
    soft_if(const SEAL_c& seal, int poly_degree) : _seal{seal} {
        if (poly_degree > 0 && (poly_degree&(poly_degree-1)) == poly_degree)
        {
            cout << "Invalid poly degree for soft_if... initialized to 8..." << endl;
            _poly_degree = 8;
        }
        else {
            _poly_degree = poly_degree;
        }

        if (_poly_degree == 8) {
            _coeffs = COEFFS8;
            _free_coeff = FREE_COEFF8;
        }
        else if (_poly_degree == 16) {
            _coeffs = COEFFS16;
            _free_coeff = FREE_COEFF16;
        }
        else if (_poly_degree == 32) {
            _coeffs = COEFFS32;
            _free_coeff = FREE_COEFF32;
        }
    }

    void operator () (Ciphertext& destination, const Ciphertext& source) const;
    void operator () (Ciphertext& ciphertext) const {
        this->operator()(ciphertext, ciphertext);
    };

    [[nodiscard]] int getPolyDegree() const { return _poly_degree; }
    [[nodiscard]] vector<double> getCoeffs() const { return _coeffs; }
    [[nodiscard]] double getFreeCoeff() const { return _free_coeff; }
};

#endif //FHE_RANDOM_FOREST_SOFT_IF_H
