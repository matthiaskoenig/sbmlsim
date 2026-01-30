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
namespace model_icg_body {

void Jy_icg_body(realtype *Jy, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my){
    switch(iy) {
        case 0:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yCre_plasma_icg, 2)) + 0.5*std::pow(-myCre_plasma_icg + yCre_plasma_icg, 2)/std::pow(sigma_yCre_plasma_icg, 2);
            break;
        case 1:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yCgi_plasma_icg, 2)) + 0.5*std::pow(-myCgi_plasma_icg + yCgi_plasma_icg, 2)/std::pow(sigma_yCgi_plasma_icg, 2);
            break;
        case 2:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yCli_plasma_icg, 2)) + 0.5*std::pow(-myCli_plasma_icg + yCli_plasma_icg, 2)/std::pow(sigma_yCli_plasma_icg, 2);
            break;
        case 3:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yClu_plasma_icg, 2)) + 0.5*std::pow(-myClu_plasma_icg + yClu_plasma_icg, 2)/std::pow(sigma_yClu_plasma_icg, 2);
            break;
        case 4:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yCve_icg, 2)) + 0.5*std::pow(-myCve_icg + yCve_icg, 2)/std::pow(sigma_yCve_icg, 2);
            break;
        case 5:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yCar_icg, 2)) + 0.5*std::pow(-myCar_icg + yCar_icg, 2)/std::pow(sigma_yCar_icg, 2);
            break;
        case 6:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yCpo_icg, 2)) + 0.5*std::pow(-myCpo_icg + yCpo_icg, 2)/std::pow(sigma_yCpo_icg, 2);
            break;
        case 7:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yChv_icg, 2)) + 0.5*std::pow(-myChv_icg + yChv_icg, 2)/std::pow(sigma_yChv_icg, 2);
            break;
        case 8:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yAfeces_icg, 2)) + 0.5*std::pow(-myAfeces_icg + yAfeces_icg, 2)/std::pow(sigma_yAfeces_icg, 2);
            break;
        case 9:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yLI__bil_ext, 2)) + 0.5*std::pow(-myLI__bil_ext + yLI__bil_ext, 2)/std::pow(sigma_yLI__bil_ext, 2);
            break;
        case 10:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yLI__icg, 2)) + 0.5*std::pow(-myLI__icg + yLI__icg, 2)/std::pow(sigma_yLI__icg, 2);
            break;
        case 11:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yLI__icg_bi, 2)) + 0.5*std::pow(-myLI__icg_bi + yLI__icg_bi, 2)/std::pow(sigma_yLI__icg_bi, 2);
            break;
        case 12:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yIVDOSE_icg, 2)) + 0.5*std::pow(-myIVDOSE_icg + yIVDOSE_icg, 2)/std::pow(sigma_yIVDOSE_icg, 2);
            break;
        case 13:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_ycum_dose_icg, 2)) + 0.5*std::pow(-mycum_dose_icg + ycum_dose_icg, 2)/std::pow(sigma_ycum_dose_icg, 2);
            break;
        case 14:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yFVre, 2)) + 0.5*std::pow(-myFVre + yFVre, 2)/std::pow(sigma_yFVre, 2);
            break;
        case 15:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yFQre, 2)) + 0.5*std::pow(-myFQre + yFQre, 2)/std::pow(sigma_yFQre, 2);
            break;
        case 16:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yBSA, 2)) + 0.5*std::pow(-myBSA + yBSA, 2)/std::pow(sigma_yBSA, 2);
            break;
        case 17:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yCO, 2)) + 0.5*std::pow(-myCO + yCO, 2)/std::pow(sigma_yCO, 2);
            break;
        case 18:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yQC, 2)) + 0.5*std::pow(-myQC + yQC, 2)/std::pow(sigma_yQC, 2);
            break;
        case 19:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yQlu, 2)) + 0.5*std::pow(-myQlu + yQlu, 2)/std::pow(sigma_yQlu, 2);
            break;
        case 20:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yQre, 2)) + 0.5*std::pow(-myQre + yQre, 2)/std::pow(sigma_yQre, 2);
            break;
        case 21:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yQh, 2)) + 0.5*std::pow(-myQh + yQh, 2)/std::pow(sigma_yQh, 2);
            break;
        case 22:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yQgi, 2)) + 0.5*std::pow(-myQgi + yQgi, 2)/std::pow(sigma_yQgi, 2);
            break;
        case 23:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yQpo, 2)) + 0.5*std::pow(-myQpo + yQpo, 2)/std::pow(sigma_yQpo, 2);
            break;
        case 24:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yQha, 2)) + 0.5*std::pow(-myQha + yQha, 2)/std::pow(sigma_yQha, 2);
            break;
        case 25:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yKi_icg, 2)) + 0.5*std::pow(-myKi_icg + yKi_icg, 2)/std::pow(sigma_yKi_icg, 2);
            break;
        case 26:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yAre_plasma_icg, 2)) + 0.5*std::pow(-myAre_plasma_icg + yAre_plasma_icg, 2)/std::pow(sigma_yAre_plasma_icg, 2);
            break;
        case 27:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yXre_plasma_icg, 2)) + 0.5*std::pow(-myXre_plasma_icg + yXre_plasma_icg, 2)/std::pow(sigma_yXre_plasma_icg, 2);
            break;
        case 28:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yMre_plasma_icg, 2)) + 0.5*std::pow(-myMre_plasma_icg + yMre_plasma_icg, 2)/std::pow(sigma_yMre_plasma_icg, 2);
            break;
        case 29:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yAgi_plasma_icg, 2)) + 0.5*std::pow(-myAgi_plasma_icg + yAgi_plasma_icg, 2)/std::pow(sigma_yAgi_plasma_icg, 2);
            break;
        case 30:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yXgi_plasma_icg, 2)) + 0.5*std::pow(-myXgi_plasma_icg + yXgi_plasma_icg, 2)/std::pow(sigma_yXgi_plasma_icg, 2);
            break;
        case 31:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yMgi_plasma_icg, 2)) + 0.5*std::pow(-myMgi_plasma_icg + yMgi_plasma_icg, 2)/std::pow(sigma_yMgi_plasma_icg, 2);
            break;
        case 32:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yAli_plasma_icg, 2)) + 0.5*std::pow(-myAli_plasma_icg + yAli_plasma_icg, 2)/std::pow(sigma_yAli_plasma_icg, 2);
            break;
        case 33:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yXli_plasma_icg, 2)) + 0.5*std::pow(-myXli_plasma_icg + yXli_plasma_icg, 2)/std::pow(sigma_yXli_plasma_icg, 2);
            break;
        case 34:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yMli_plasma_icg, 2)) + 0.5*std::pow(-myMli_plasma_icg + yMli_plasma_icg, 2)/std::pow(sigma_yMli_plasma_icg, 2);
            break;
        case 35:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yAlu_plasma_icg, 2)) + 0.5*std::pow(-myAlu_plasma_icg + yAlu_plasma_icg, 2)/std::pow(sigma_yAlu_plasma_icg, 2);
            break;
        case 36:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yXlu_plasma_icg, 2)) + 0.5*std::pow(-myXlu_plasma_icg + yXlu_plasma_icg, 2)/std::pow(sigma_yXlu_plasma_icg, 2);
            break;
        case 37:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yMlu_plasma_icg, 2)) + 0.5*std::pow(-myMlu_plasma_icg + yMlu_plasma_icg, 2)/std::pow(sigma_yMlu_plasma_icg, 2);
            break;
        case 38:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yAve_icg, 2)) + 0.5*std::pow(-myAve_icg + yAve_icg, 2)/std::pow(sigma_yAve_icg, 2);
            break;
        case 39:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yXve_icg, 2)) + 0.5*std::pow(-myXve_icg + yXve_icg, 2)/std::pow(sigma_yXve_icg, 2);
            break;
        case 40:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yMve_icg, 2)) + 0.5*std::pow(-myMve_icg + yMve_icg, 2)/std::pow(sigma_yMve_icg, 2);
            break;
        case 41:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yAar_icg, 2)) + 0.5*std::pow(-myAar_icg + yAar_icg, 2)/std::pow(sigma_yAar_icg, 2);
            break;
        case 42:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yXar_icg, 2)) + 0.5*std::pow(-myXar_icg + yXar_icg, 2)/std::pow(sigma_yXar_icg, 2);
            break;
        case 43:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yMar_icg, 2)) + 0.5*std::pow(-myMar_icg + yMar_icg, 2)/std::pow(sigma_yMar_icg, 2);
            break;
        case 44:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yApo_icg, 2)) + 0.5*std::pow(-myApo_icg + yApo_icg, 2)/std::pow(sigma_yApo_icg, 2);
            break;
        case 45:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yXpo_icg, 2)) + 0.5*std::pow(-myXpo_icg + yXpo_icg, 2)/std::pow(sigma_yXpo_icg, 2);
            break;
        case 46:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yMpo_icg, 2)) + 0.5*std::pow(-myMpo_icg + yMpo_icg, 2)/std::pow(sigma_yMpo_icg, 2);
            break;
        case 47:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yAhv_icg, 2)) + 0.5*std::pow(-myAhv_icg + yAhv_icg, 2)/std::pow(sigma_yAhv_icg, 2);
            break;
        case 48:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yXhv_icg, 2)) + 0.5*std::pow(-myXhv_icg + yXhv_icg, 2)/std::pow(sigma_yXhv_icg, 2);
            break;
        case 49:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yMhv_icg, 2)) + 0.5*std::pow(-myMhv_icg + yMhv_icg, 2)/std::pow(sigma_yMhv_icg, 2);
            break;
        case 50:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yER_icg, 2)) + 0.5*std::pow(-myER_icg + yER_icg, 2)/std::pow(sigma_yER_icg, 2);
            break;
        case 51:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yCLinfusion_icg, 2)) + 0.5*std::pow(-myCLinfusion_icg + yCLinfusion_icg, 2)/std::pow(sigma_yCLinfusion_icg, 2);
            break;
        case 52:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVbi, 2)) + 0.5*std::pow(-myVbi + yVbi, 2)/std::pow(sigma_yVbi, 2);
            break;
        case 53:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVgi, 2)) + 0.5*std::pow(-myVgi + yVgi, 2)/std::pow(sigma_yVgi, 2);
            break;
        case 54:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVli, 2)) + 0.5*std::pow(-myVli + yVli, 2)/std::pow(sigma_yVli, 2);
            break;
        case 55:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVlu, 2)) + 0.5*std::pow(-myVlu + yVlu, 2)/std::pow(sigma_yVlu, 2);
            break;
        case 56:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVre, 2)) + 0.5*std::pow(-myVre + yVre, 2)/std::pow(sigma_yVre, 2);
            break;
        case 57:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVve, 2)) + 0.5*std::pow(-myVve + yVve, 2)/std::pow(sigma_yVve, 2);
            break;
        case 58:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVar, 2)) + 0.5*std::pow(-myVar + yVar, 2)/std::pow(sigma_yVar, 2);
            break;
        case 59:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVpo, 2)) + 0.5*std::pow(-myVpo + yVpo, 2)/std::pow(sigma_yVpo, 2);
            break;
        case 60:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVhv, 2)) + 0.5*std::pow(-myVhv + yVhv, 2)/std::pow(sigma_yVhv, 2);
            break;
        case 61:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVre_plasma, 2)) + 0.5*std::pow(-myVre_plasma + yVre_plasma, 2)/std::pow(sigma_yVre_plasma, 2);
            break;
        case 62:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVre_tissue, 2)) + 0.5*std::pow(-myVre_tissue + yVre_tissue, 2)/std::pow(sigma_yVre_tissue, 2);
            break;
        case 63:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVgi_plasma, 2)) + 0.5*std::pow(-myVgi_plasma + yVgi_plasma, 2)/std::pow(sigma_yVgi_plasma, 2);
            break;
        case 64:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVgi_tissue, 2)) + 0.5*std::pow(-myVgi_tissue + yVgi_tissue, 2)/std::pow(sigma_yVgi_tissue, 2);
            break;
        case 65:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVli_plasma, 2)) + 0.5*std::pow(-myVli_plasma + yVli_plasma, 2)/std::pow(sigma_yVli_plasma, 2);
            break;
        case 66:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVli_tissue, 2)) + 0.5*std::pow(-myVli_tissue + yVli_tissue, 2)/std::pow(sigma_yVli_tissue, 2);
            break;
        case 67:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVlu_plasma, 2)) + 0.5*std::pow(-myVlu_plasma + yVlu_plasma, 2)/std::pow(sigma_yVlu_plasma, 2);
            break;
        case 68:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVlu_tissue, 2)) + 0.5*std::pow(-myVlu_tissue + yVlu_tissue, 2)/std::pow(sigma_yVlu_tissue, 2);
            break;
        case 69:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_yVfeces, 2)) + 0.5*std::pow(-myVfeces + yVfeces, 2)/std::pow(sigma_yVfeces, 2);
            break;
    }
}

} // namespace model_icg_body
} // namespace amici
