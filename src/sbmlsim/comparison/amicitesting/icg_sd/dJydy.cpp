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
namespace model_icg_sd {

void dJydy_icg_sd(realtype *dJydy, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my){
    switch(iy) {
        case 0:
            dJydy[0] = (-1.0*myCre_plasma_icg + 1.0*yCre_plasma_icg)/std::pow(sigma_yCre_plasma_icg, 2);
            break;
        case 1:
            dJydy[0] = (-1.0*myCgi_plasma_icg + 1.0*yCgi_plasma_icg)/std::pow(sigma_yCgi_plasma_icg, 2);
            break;
        case 2:
            dJydy[0] = (-1.0*myCli_plasma_icg + 1.0*yCli_plasma_icg)/std::pow(sigma_yCli_plasma_icg, 2);
            break;
        case 3:
            dJydy[0] = (-1.0*myClu_plasma_icg + 1.0*yClu_plasma_icg)/std::pow(sigma_yClu_plasma_icg, 2);
            break;
        case 4:
            dJydy[0] = (-1.0*myCve_icg + 1.0*yCve_icg)/std::pow(sigma_yCve_icg, 2);
            break;
        case 5:
            dJydy[0] = (-1.0*myCar_icg + 1.0*yCar_icg)/std::pow(sigma_yCar_icg, 2);
            break;
        case 6:
            dJydy[0] = (-1.0*myCpo_icg + 1.0*yCpo_icg)/std::pow(sigma_yCpo_icg, 2);
            break;
        case 7:
            dJydy[0] = (-1.0*myChv_icg + 1.0*yChv_icg)/std::pow(sigma_yChv_icg, 2);
            break;
        case 8:
            dJydy[0] = (-1.0*myAfeces_icg + 1.0*yAfeces_icg)/std::pow(sigma_yAfeces_icg, 2);
            break;
        case 9:
            dJydy[0] = (-1.0*myLI__bil_ext + 1.0*yLI__bil_ext)/std::pow(sigma_yLI__bil_ext, 2);
            break;
        case 10:
            dJydy[0] = (-1.0*myLI__icg + 1.0*yLI__icg)/std::pow(sigma_yLI__icg, 2);
            break;
        case 11:
            dJydy[0] = (-1.0*myLI__icg_bi + 1.0*yLI__icg_bi)/std::pow(sigma_yLI__icg_bi, 2);
            break;
        case 12:
            dJydy[0] = (-1.0*myIVDOSE_icg + 1.0*yIVDOSE_icg)/std::pow(sigma_yIVDOSE_icg, 2);
            break;
        case 13:
            dJydy[0] = (-1.0*mycum_dose_icg + 1.0*ycum_dose_icg)/std::pow(sigma_ycum_dose_icg, 2);
            break;
        case 14:
            dJydy[0] = (-1.0*myFVre + 1.0*yFVre)/std::pow(sigma_yFVre, 2);
            break;
        case 15:
            dJydy[0] = (-1.0*myFQre + 1.0*yFQre)/std::pow(sigma_yFQre, 2);
            break;
        case 16:
            dJydy[0] = (-1.0*myBSA + 1.0*yBSA)/std::pow(sigma_yBSA, 2);
            break;
        case 17:
            dJydy[0] = (-1.0*myCO + 1.0*yCO)/std::pow(sigma_yCO, 2);
            break;
        case 18:
            dJydy[0] = (-1.0*myQC + 1.0*yQC)/std::pow(sigma_yQC, 2);
            break;
        case 19:
            dJydy[0] = (-1.0*myQlu + 1.0*yQlu)/std::pow(sigma_yQlu, 2);
            break;
        case 20:
            dJydy[0] = (-1.0*myQre + 1.0*yQre)/std::pow(sigma_yQre, 2);
            break;
        case 21:
            dJydy[0] = (-1.0*myQh + 1.0*yQh)/std::pow(sigma_yQh, 2);
            break;
        case 22:
            dJydy[0] = (-1.0*myQgi + 1.0*yQgi)/std::pow(sigma_yQgi, 2);
            break;
        case 23:
            dJydy[0] = (-1.0*myQpo + 1.0*yQpo)/std::pow(sigma_yQpo, 2);
            break;
        case 24:
            dJydy[0] = (-1.0*myQha + 1.0*yQha)/std::pow(sigma_yQha, 2);
            break;
        case 25:
            dJydy[0] = (-1.0*myKi_icg + 1.0*yKi_icg)/std::pow(sigma_yKi_icg, 2);
            break;
        case 26:
            dJydy[0] = (-1.0*myAre_plasma_icg + 1.0*yAre_plasma_icg)/std::pow(sigma_yAre_plasma_icg, 2);
            break;
        case 27:
            dJydy[0] = (-1.0*myXre_plasma_icg + 1.0*yXre_plasma_icg)/std::pow(sigma_yXre_plasma_icg, 2);
            break;
        case 28:
            dJydy[0] = (-1.0*myMre_plasma_icg + 1.0*yMre_plasma_icg)/std::pow(sigma_yMre_plasma_icg, 2);
            break;
        case 29:
            dJydy[0] = (-1.0*myAgi_plasma_icg + 1.0*yAgi_plasma_icg)/std::pow(sigma_yAgi_plasma_icg, 2);
            break;
        case 30:
            dJydy[0] = (-1.0*myXgi_plasma_icg + 1.0*yXgi_plasma_icg)/std::pow(sigma_yXgi_plasma_icg, 2);
            break;
        case 31:
            dJydy[0] = (-1.0*myMgi_plasma_icg + 1.0*yMgi_plasma_icg)/std::pow(sigma_yMgi_plasma_icg, 2);
            break;
        case 32:
            dJydy[0] = (-1.0*myAli_plasma_icg + 1.0*yAli_plasma_icg)/std::pow(sigma_yAli_plasma_icg, 2);
            break;
        case 33:
            dJydy[0] = (-1.0*myXli_plasma_icg + 1.0*yXli_plasma_icg)/std::pow(sigma_yXli_plasma_icg, 2);
            break;
        case 34:
            dJydy[0] = (-1.0*myMli_plasma_icg + 1.0*yMli_plasma_icg)/std::pow(sigma_yMli_plasma_icg, 2);
            break;
        case 35:
            dJydy[0] = (-1.0*myAlu_plasma_icg + 1.0*yAlu_plasma_icg)/std::pow(sigma_yAlu_plasma_icg, 2);
            break;
        case 36:
            dJydy[0] = (-1.0*myXlu_plasma_icg + 1.0*yXlu_plasma_icg)/std::pow(sigma_yXlu_plasma_icg, 2);
            break;
        case 37:
            dJydy[0] = (-1.0*myMlu_plasma_icg + 1.0*yMlu_plasma_icg)/std::pow(sigma_yMlu_plasma_icg, 2);
            break;
        case 38:
            dJydy[0] = (-1.0*myAve_icg + 1.0*yAve_icg)/std::pow(sigma_yAve_icg, 2);
            break;
        case 39:
            dJydy[0] = (-1.0*myXve_icg + 1.0*yXve_icg)/std::pow(sigma_yXve_icg, 2);
            break;
        case 40:
            dJydy[0] = (-1.0*myMve_icg + 1.0*yMve_icg)/std::pow(sigma_yMve_icg, 2);
            break;
        case 41:
            dJydy[0] = (-1.0*myAar_icg + 1.0*yAar_icg)/std::pow(sigma_yAar_icg, 2);
            break;
        case 42:
            dJydy[0] = (-1.0*myXar_icg + 1.0*yXar_icg)/std::pow(sigma_yXar_icg, 2);
            break;
        case 43:
            dJydy[0] = (-1.0*myMar_icg + 1.0*yMar_icg)/std::pow(sigma_yMar_icg, 2);
            break;
        case 44:
            dJydy[0] = (-1.0*myApo_icg + 1.0*yApo_icg)/std::pow(sigma_yApo_icg, 2);
            break;
        case 45:
            dJydy[0] = (-1.0*myXpo_icg + 1.0*yXpo_icg)/std::pow(sigma_yXpo_icg, 2);
            break;
        case 46:
            dJydy[0] = (-1.0*myMpo_icg + 1.0*yMpo_icg)/std::pow(sigma_yMpo_icg, 2);
            break;
        case 47:
            dJydy[0] = (-1.0*myAhv_icg + 1.0*yAhv_icg)/std::pow(sigma_yAhv_icg, 2);
            break;
        case 48:
            dJydy[0] = (-1.0*myXhv_icg + 1.0*yXhv_icg)/std::pow(sigma_yXhv_icg, 2);
            break;
        case 49:
            dJydy[0] = (-1.0*myMhv_icg + 1.0*yMhv_icg)/std::pow(sigma_yMhv_icg, 2);
            break;
        case 50:
            dJydy[0] = (-1.0*myER_icg + 1.0*yER_icg)/std::pow(sigma_yER_icg, 2);
            break;
        case 51:
            dJydy[0] = (-1.0*myCLinfusion_icg + 1.0*yCLinfusion_icg)/std::pow(sigma_yCLinfusion_icg, 2);
            break;
        case 52:
            dJydy[0] = (-1.0*myVbi + 1.0*yVbi)/std::pow(sigma_yVbi, 2);
            break;
        case 53:
            dJydy[0] = (-1.0*myVgi + 1.0*yVgi)/std::pow(sigma_yVgi, 2);
            break;
        case 54:
            dJydy[0] = (-1.0*myVli + 1.0*yVli)/std::pow(sigma_yVli, 2);
            break;
        case 55:
            dJydy[0] = (-1.0*myVlu + 1.0*yVlu)/std::pow(sigma_yVlu, 2);
            break;
        case 56:
            dJydy[0] = (-1.0*myVre + 1.0*yVre)/std::pow(sigma_yVre, 2);
            break;
        case 57:
            dJydy[0] = (-1.0*myVve + 1.0*yVve)/std::pow(sigma_yVve, 2);
            break;
        case 58:
            dJydy[0] = (-1.0*myVar + 1.0*yVar)/std::pow(sigma_yVar, 2);
            break;
        case 59:
            dJydy[0] = (-1.0*myVpo + 1.0*yVpo)/std::pow(sigma_yVpo, 2);
            break;
        case 60:
            dJydy[0] = (-1.0*myVhv + 1.0*yVhv)/std::pow(sigma_yVhv, 2);
            break;
        case 61:
            dJydy[0] = (-1.0*myVre_plasma + 1.0*yVre_plasma)/std::pow(sigma_yVre_plasma, 2);
            break;
        case 62:
            dJydy[0] = (-1.0*myVre_tissue + 1.0*yVre_tissue)/std::pow(sigma_yVre_tissue, 2);
            break;
        case 63:
            dJydy[0] = (-1.0*myVgi_plasma + 1.0*yVgi_plasma)/std::pow(sigma_yVgi_plasma, 2);
            break;
        case 64:
            dJydy[0] = (-1.0*myVgi_tissue + 1.0*yVgi_tissue)/std::pow(sigma_yVgi_tissue, 2);
            break;
        case 65:
            dJydy[0] = (-1.0*myVli_plasma + 1.0*yVli_plasma)/std::pow(sigma_yVli_plasma, 2);
            break;
        case 66:
            dJydy[0] = (-1.0*myVli_tissue + 1.0*yVli_tissue)/std::pow(sigma_yVli_tissue, 2);
            break;
        case 67:
            dJydy[0] = (-1.0*myVlu_plasma + 1.0*yVlu_plasma)/std::pow(sigma_yVlu_plasma, 2);
            break;
        case 68:
            dJydy[0] = (-1.0*myVlu_tissue + 1.0*yVlu_tissue)/std::pow(sigma_yVlu_tissue, 2);
            break;
        case 69:
            dJydy[0] = (-1.0*myVfeces + 1.0*yVfeces)/std::pow(sigma_yVfeces, 2);
            break;
    }
}

} // namespace model_icg_sd
} // namespace amici
