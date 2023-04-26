#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "p.h"
#include "y.h"
#include "sigmay.h"
#include "my.h"

namespace amici {
namespace model_icg_liver {

void dJydsigma_icg_liver(realtype *dJydsigma, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my){
    switch(iy) {
        case 0:
            dJydsigma[0] = 1.0/sigma_yicg_ext - 1.0*std::pow(-myicg_ext + yicg_ext, 2)/std::pow(sigma_yicg_ext, 3);
            break;
        case 1:
            dJydsigma[1] = 1.0/sigma_ybil_ext - 1.0*std::pow(-mybil_ext + ybil_ext, 2)/std::pow(sigma_ybil_ext, 3);
            break;
        case 2:
            dJydsigma[2] = 1.0/sigma_yicg - 1.0*std::pow(-myicg + yicg, 2)/std::pow(sigma_yicg, 3);
            break;
        case 3:
            dJydsigma[3] = 1.0/sigma_yicg_bi - 1.0*std::pow(-myicg_bi + yicg_bi, 2)/std::pow(sigma_yicg_bi, 3);
            break;
        case 4:
            dJydsigma[4] = 1.0/sigma_yicg_feces - 1.0*std::pow(-myicg_feces + yicg_feces, 2)/std::pow(sigma_yicg_feces, 3);
            break;
        case 5:
            dJydsigma[5] = 1.0/sigma_yVext - 1.0*std::pow(-myVext + yVext, 2)/std::pow(sigma_yVext, 3);
            break;
        case 6:
            dJydsigma[6] = 1.0/sigma_yVli - 1.0*std::pow(-myVli + yVli, 2)/std::pow(sigma_yVli, 3);
            break;
        case 7:
            dJydsigma[7] = 1.0/sigma_yVbi - 1.0*std::pow(-myVbi + yVbi, 2)/std::pow(sigma_yVbi, 3);
            break;
        case 8:
            dJydsigma[8] = 1.0/sigma_yVfeces - 1.0*std::pow(-myVfeces + yVfeces, 2)/std::pow(sigma_yVfeces, 3);
            break;
    }
}

} // namespace model_icg_liver
} // namespace amici
