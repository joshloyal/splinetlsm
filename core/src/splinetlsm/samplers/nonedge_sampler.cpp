#include <vector>

#include "splinetlsm.h"


namespace splinetlsm {
    
    arma::uvec find_edge_ids(const arma::sp_mat& Y, uint col_id) {
        std::vector<arma::uword> edge_ids;

        // loop through columns (faster for csc format)
        for (auto j = Y.begin_col(col_id); j != Y.end_col(col_id); ++j) {
            edge_ids.push_back(j.row());
        }
 
        // hack to avoid picking self-loops when sampling non-edges
        edge_ids.push_back(col_id);

        return arma::uvec(edge_ids);
    }

    inline int get_num_nonedges(int n_nodes, int n_edges, double proportion) {
        return ((n_nodes - 1 - n_edges) < std::floor(proportion * n_edges) ? 
            (n_nodes - 1 - n_edges) : std::floor(proportion * n_edges));
    }

    NonEdgeSampler::NonEdgeSampler(double proportion) :
        proportion_(proportion) {}

    std::tuple<arma::field<arma::uvec>, arma::mat, arma::mat>
    NonEdgeSampler::draw(const sp_cube& Y, const arma::uvec& time_indices) {
        uint n_time_steps = time_indices.n_elem;
        uint n_nodes = Y(0).n_rows;

        arma::field<arma::uvec> dyad_subsamples(n_time_steps, n_nodes); 
        arma::mat degrees(n_time_steps, n_nodes);
        arma::mat weights(n_time_steps, n_nodes);
        for (uint t = 0; t < n_time_steps; ++t) {
            for (uint i = 0; i < n_nodes; ++i) {
             
                // edges
                arma::uvec edge_ids = find_edge_ids(Y(time_indices(t)), i);
                int n_edges = edge_ids.n_rows - 1;  // do not count self-loops
                degrees(t, i) = n_edges;

                // non-edges
                int n_nonedges = get_num_nonedges(n_nodes, n_edges, proportion_);
                arma::uvec nonedge_ids = arma::regspace<arma::uvec>(
                        0, n_nodes - 1);
                nonedge_ids.shed_rows(edge_ids);
                
                // uniformely sample non-edges
                arma::uvec subsample = nonedge_ids.elem(
                    arma::randperm(nonedge_ids.n_elem, n_nonedges));

                // store combined result
                dyad_subsamples(t, i) = join_cols(edge_ids, subsample);
                weights(t, i) = nonedge_ids.n_elem / n_nonedges;
            }
        }
        
        return {dyad_subsamples, degrees, weights};
    }
}
