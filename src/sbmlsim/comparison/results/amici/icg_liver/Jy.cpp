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

void Jy_icg_liver(realtype *Jy, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my){
    switch(iy) {
        case 0:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yicg_ext, 2)) + 0.5*std::pow(-myicg_ext + yicg_ext, 2)/std::pow(sigma_yicg_ext, 2);
            break;
        case 1:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_ybil_ext, 2)) + 0.5*std::pow(-mybil_ext + ybil_ext, 2)/std::pow(sigma_ybil_ext, 2);
            break;
        case 2:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yicg, 2)) + 0.5*std::pow(-myicg + yicg, 2)/std::pow(sigma_yicg, 2);
            break;
        case 3:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yicg_bi, 2)) + 0.5*std::pow(-myicg_bi + yicg_bi, 2)/std::pow(sigma_yicg_bi, 2);
            break;
        case 4:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yicg_feces, 2)) + 0.5*std::pow(-myicg_feces + yicg_feces, 2)/std::pow(sigma_yicg_feces, 2);
            break;
        case 5:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVext, 2)) + 0.5*std::pow(-myVext + yVext, 2)/std::pow(sigma_yVext, 2);
            break;
        case 6:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVli, 2)) + 0.5*std::pow(-myVli + yVli, 2)/std::pow(sigma_yVli, 2);
            break;
        case 7:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVbi, 2)) + 0.5*std::pow(-myVbi + yVbi, 2)/std::pow(sigma_yVbi, 2);
            break;
        case 8:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVfeces, 2)) + 0.5*std::pow(-myVfeces + yVfeces, 2)/std::pow(sigma_yVfeces, 2);
            break;
    }
}

} // namespace model_icg_liver
} // namespace amici
