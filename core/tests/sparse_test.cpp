#include "asserts.h"
#include <catch2/catch_test_macros.hpp>

#include "splinetlsm.h"


TEST_CASE("Sparse Matrix", "[sparse]") {
    arma::arma_rng::set_seed(123);

    //arma::sp_mat A = arma::sprandu<arma::sp_mat>(1000, 1000, 0.1);
    //arma::sp_mat B = arma::sprandn(100, 10, 0.1);
    arma::sp_mat B = arma::speye(100,100);
    arma::umat locations = { { 1, 7, 9 },
                             { 2, 8, 9 } };

    arma::vec values = { 1.0, 2.0, 3.0 };
    arma::sp_mat out = B * B;
    arma::sp_mat X(locations, values);
    //arma::mat A = arma::randu(10, 100);
    //arma::sp_mat A = arma::speye(5,5);
    //std::cout << out << std::endl;
}
