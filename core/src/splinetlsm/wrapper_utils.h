#pragma once

namespace splinetlsm {
    void set_spcube_value(arma::field<arma::sp_mat>& A, arma::sp_mat& B, uint index);
    void set_4darray_value(arma::field<arma::cube>& A, arma::cube& B, uint index);
}
