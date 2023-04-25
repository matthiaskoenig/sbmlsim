#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "p.h"
#include "y.h"
#include "sigmay.h"

namespace amici {
namespace model_icg_body {

void sigmay_icg_body(realtype *sigmay, const realtype t, const realtype *p, const realtype *k, const realtype *y){
    sigma_yCre_plasma_icg = 1.0;  // sigmay[0]
    sigma_yCgi_plasma_icg = 1.0;  // sigmay[1]
    sigma_yCli_plasma_icg = 1.0;  // sigmay[2]
    sigma_yClu_plasma_icg = 1.0;  // sigmay[3]
    sigma_yCve_icg = 1.0;  // sigmay[4]
    sigma_yCar_icg = 1.0;  // sigmay[5]
    sigma_yCpo_icg = 1.0;  // sigmay[6]
    sigma_yChv_icg = 1.0;  // sigmay[7]
    sigma_yAfeces_icg = 1.0;  // sigmay[8]
    sigma_yLI__bil_ext = 1.0;  // sigmay[9]
    sigma_yLI__icg = 1.0;  // sigmay[10]
    sigma_yLI__icg_bi = 1.0;  // sigmay[11]
    sigma_yIVDOSE_icg = 1.0;  // sigmay[12]
    sigma_ycum_dose_icg = 1.0;  // sigmay[13]
    sigma_yFVre = 1.0;  // sigmay[14]
    sigma_yFQre = 1.0;  // sigmay[15]
    sigma_yBSA = 1.0;  // sigmay[16]
    sigma_yCO = 1.0;  // sigmay[17]
    sigma_yQC = 1.0;  // sigmay[18]
    sigma_yQlu = 1.0;  // sigmay[19]
    sigma_yQre = 1.0;  // sigmay[20]
    sigma_yQh = 1.0;  // sigmay[21]
    sigma_yQgi = 1.0;  // sigmay[22]
    sigma_yQpo = 1.0;  // sigmay[23]
    sigma_yQha = 1.0;  // sigmay[24]
    sigma_yKi_icg = 1.0;  // sigmay[25]
    sigma_yAre_plasma_icg = 1.0;  // sigmay[26]
    sigma_yXre_plasma_icg = 1.0;  // sigmay[27]
    sigma_yMre_plasma_icg = 1.0;  // sigmay[28]
    sigma_yAgi_plasma_icg = 1.0;  // sigmay[29]
    sigma_yXgi_plasma_icg = 1.0;  // sigmay[30]
    sigma_yMgi_plasma_icg = 1.0;  // sigmay[31]
    sigma_yAli_plasma_icg = 1.0;  // sigmay[32]
    sigma_yXli_plasma_icg = 1.0;  // sigmay[33]
    sigma_yMli_plasma_icg = 1.0;  // sigmay[34]
    sigma_yAlu_plasma_icg = 1.0;  // sigmay[35]
    sigma_yXlu_plasma_icg = 1.0;  // sigmay[36]
    sigma_yMlu_plasma_icg = 1.0;  // sigmay[37]
    sigma_yAve_icg = 1.0;  // sigmay[38]
    sigma_yXve_icg = 1.0;  // sigmay[39]
    sigma_yMve_icg = 1.0;  // sigmay[40]
    sigma_yAar_icg = 1.0;  // sigmay[41]
    sigma_yXar_icg = 1.0;  // sigmay[42]
    sigma_yMar_icg = 1.0;  // sigmay[43]
    sigma_yApo_icg = 1.0;  // sigmay[44]
    sigma_yXpo_icg = 1.0;  // sigmay[45]
    sigma_yMpo_icg = 1.0;  // sigmay[46]
    sigma_yAhv_icg = 1.0;  // sigmay[47]
    sigma_yXhv_icg = 1.0;  // sigmay[48]
    sigma_yMhv_icg = 1.0;  // sigmay[49]
    sigma_yER_icg = 1.0;  // sigmay[50]
    sigma_yCLinfusion_icg = 1.0;  // sigmay[51]
    sigma_yVbi = 1.0;  // sigmay[52]
    sigma_yVgi = 1.0;  // sigmay[53]
    sigma_yVli = 1.0;  // sigmay[54]
    sigma_yVlu = 1.0;  // sigmay[55]
    sigma_yVre = 1.0;  // sigmay[56]
    sigma_yVve = 1.0;  // sigmay[57]
    sigma_yVar = 1.0;  // sigmay[58]
    sigma_yVpo = 1.0;  // sigmay[59]
    sigma_yVhv = 1.0;  // sigmay[60]
    sigma_yVre_plasma = 1.0;  // sigmay[61]
    sigma_yVre_tissue = 1.0;  // sigmay[62]
    sigma_yVgi_plasma = 1.0;  // sigmay[63]
    sigma_yVgi_tissue = 1.0;  // sigmay[64]
    sigma_yVli_plasma = 1.0;  // sigmay[65]
    sigma_yVli_tissue = 1.0;  // sigmay[66]
    sigma_yVlu_plasma = 1.0;  // sigmay[67]
    sigma_yVlu_tissue = 1.0;  // sigmay[68]
    sigma_yVfeces = 1.0;  // sigmay[69]
}

} // namespace model_icg_body
} // namespace amici
