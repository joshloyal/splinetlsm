#pragma once

//#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>

#include "splinetlsm/typedefs.h"

#include "splinetlsm/samplers/time_sampler.h"
#include "splinetlsm/samplers/nonedge_sampler.h"
#include "splinetlsm/samplers/dyad_sampler.h"

#include "splinetlsm/config.h"
#include "splinetlsm/parameters.h"
#include "splinetlsm/moments.h"
#include "splinetlsm/omega.h"
#include "splinetlsm/latent_positions.h"
#include "splinetlsm/coefs.h"
#include "splinetlsm/variances.h"
#include "splinetlsm/svi.h"

#include "splinetlsm/wrapper_utils.h"
