#include <vector>

#include "drforest.h"


namespace splinetlsm {
    
    arma::uvec find_edge_ids(arma::sp_mat& Y, uint col_id) {
        std::vector<uint> edge_ids;

        // loop through columns (faster for csc format)
        auto start = Y.begin_col(col_id);
        auto end = Y.end_col(col_id);
        for (auto j = start; j != end; ++j) {
            edge_ids.push_back(j.col());
        }
        
        // hack to avoid picking self-loops when sampling non-edges
        edge_ids.push_back(col_id);

        return arma::uvec(edge_ids);
    }

    inline int get_num_nonedges(int n_nodes, int n_edges, double proportion) {
        return ((n_nodes - 1 - n_edges) < std::floor(proportion * n_edges) ? 
            (n_nodes - 1 - n_edges) : std::floor(proportion * n_edges));
    }

    NonEdgeSampler::NonEdgeampler(double proportion) :
        proportion_(proportion) {}

    std::tuple<arma::field<arma::uvec>, arma::field<arma::uvec>, arma::field<arma::vec> >
    NonEdgeSampler::draw(
            const sp_cube& Y, arma::uvec& time_indices) {
        uint n_time_steps = time_indices.n_elem;
        uint n_nodes = Y.n_slices;

        arma::field<arma::uvec> dyad_subsample(n_time_steps, n_nodes); 
        arma::mat degrees(n_time_steps, n_nodes);
        arma::mat weights(n_time_steps, n_nodes);
        for (auto t : time_indices) {
            for (int i = 0; i < n_nodes; ++i) {
                arma::uvec subsample;
                
                // edges
                arma::uvec edge_ids = find_edge_ids(Y(t), i);
                int n_edges = edge_ids.n_rows - 1;  // do not count self-loops
                degrees(t, i) = n_edges;

                // non-edges
                int n_nonedges = get_num_nonedges(n_nodes, n_edges, proportion_);
                arma::uvec nonedge_ids = arma::regspace<arma::uvec>(
                        0, n_nodes - 1).shed_rows(edge_ids);
                
                // uniformely sample non-edges
                arma::uvec subsample;
                subsample.resize(n_nonedges);
                subsample = nonedge_ids.elem(
                        arma::randomperm(nonedge_ids.size(), n_nonedges))

                // store combined result
                dyad_subsample(t, i) = join_cols(edge_ids, subsample);
                weights(t, i) = nonedge_ids.size() / n_nonedges 
            }
        }
        
        return {dyad_subsamples, degrees, weights};
    }
}
