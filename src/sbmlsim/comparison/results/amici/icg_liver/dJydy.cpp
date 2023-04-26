#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "p.h"
#include "y.h"
#include "sigmay.h"
#include "my.h"
#include "dJydy.h"

namespace amici {
namespace model_icg_liver {

void dJydy_icg_liver(realtype *dJydy, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my){
    switch(iy) {
        case 0:
            dJydy[0] = (-1.0*myicg_ext + 1.0*yicg_ext)/std::pow(sigma_yicg_ext, 2);
            break;
        case 1:
            dJydy[0] = (-1.0*mybil_ext + 1.0*ybil_ext)/std::pow(sigma_ybil_ext, 2);
            break;
        case 2:
            dJydy[0] = (-1.0*myicg + 1.0*yicg)/std::pow(sigma_yicg, 2);
            break;
        case 3:
            dJydy[0] = (-1.0*myicg_bi + 1.0*yicg_bi)/std::pow(sigma_yicg_bi, 2);
            break;
        case 4:
            dJydy[0] = (-1.0*myicg_feces + 1.0*yicg_feces)/std::pow(sigma_yicg_feces, 2);
            break;
        case 5:
            dJydy[0] = (-1.0*myVext + 1.0*yVext)/std::pow(sigma_yVext, 2);
            break;
        case 6:
            dJydy[0] = (-1.0*myVli + 1.0*yVli)/std::pow(sigma_yVli, 2);
            break;
        case 7:
            dJydy[0] = (-1.0*myVbi + 1.0*yVbi)/std::pow(sigma_yVbi, 2);
            break;
        case 8:
            dJydy[0] = (-1.0*myVfeces + 1.0*yVfeces)/std::pow(sigma_yVfeces, 2);
            break;
    }
}

} // namespace model_icg_liver
} // namespace amici
