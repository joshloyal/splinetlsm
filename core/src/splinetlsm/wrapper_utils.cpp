#include "splinetlsm.h"

namespace splinetlsm {
    
    void set_spcube_value(arma::field<arma::sp_mat>& A, arma::sp_mat& B, uint index) {
        A(index) = B;
    }
    
    void set_4darray_value(arma::field<arma::cube>& A, arma::cube& B, uint index) {
        A(index) = B;
    }

}
