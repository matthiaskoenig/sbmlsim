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
namespace model_icg_sd {

void dJydsigma_icg_sd(realtype *dJydsigma, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my){
    switch(iy) {
        case 0:
            dJydsigma[0] = 1.0/sigma_yCre_plasma_icg - 1.0*std::pow(-myCre_plasma_icg + yCre_plasma_icg, 2)/std::pow(sigma_yCre_plasma_icg, 3);
            break;
        case 1:
            dJydsigma[1] = 1.0/sigma_yCgi_plasma_icg - 1.0*std::pow(-myCgi_plasma_icg + yCgi_plasma_icg, 2)/std::pow(sigma_yCgi_plasma_icg, 3);
            break;
        case 2:
            dJydsigma[2] = 1.0/sigma_yCli_plasma_icg - 1.0*std::pow(-myCli_plasma_icg + yCli_plasma_icg, 2)/std::pow(sigma_yCli_plasma_icg, 3);
            break;
        case 3:
            dJydsigma[3] = 1.0/sigma_yClu_plasma_icg - 1.0*std::pow(-myClu_plasma_icg + yClu_plasma_icg, 2)/std::pow(sigma_yClu_plasma_icg, 3);
            break;
        case 4:
            dJydsigma[4] = 1.0/sigma_yCve_icg - 1.0*std::pow(-myCve_icg + yCve_icg, 2)/std::pow(sigma_yCve_icg, 3);
            break;
        case 5:
            dJydsigma[5] = 1.0/sigma_yCar_icg - 1.0*std::pow(-myCar_icg + yCar_icg, 2)/std::pow(sigma_yCar_icg, 3);
            break;
        case 6:
            dJydsigma[6] = 1.0/sigma_yCpo_icg - 1.0*std::pow(-myCpo_icg + yCpo_icg, 2)/std::pow(sigma_yCpo_icg, 3);
            break;
        case 7:
            dJydsigma[7] = 1.0/sigma_yChv_icg - 1.0*std::pow(-myChv_icg + yChv_icg, 2)/std::pow(sigma_yChv_icg, 3);
            break;
        case 8:
            dJydsigma[8] = 1.0/sigma_yAfeces_icg - 1.0*std::pow(-myAfeces_icg + yAfeces_icg, 2)/std::pow(sigma_yAfeces_icg, 3);
            break;
        case 9:
            dJydsigma[9] = 1.0/sigma_yLI__bil_ext - 1.0*std::pow(-myLI__bil_ext + yLI__bil_ext, 2)/std::pow(sigma_yLI__bil_ext, 3);
            break;
        case 10:
            dJydsigma[10] = 1.0/sigma_yLI__icg - 1.0*std::pow(-myLI__icg + yLI__icg, 2)/std::pow(sigma_yLI__icg, 3);
            break;
        case 11:
            dJydsigma[11] = 1.0/sigma_yLI__icg_bi - 1.0*std::pow(-myLI__icg_bi + yLI__icg_bi, 2)/std::pow(sigma_yLI__icg_bi, 3);
            break;
        case 12:
            dJydsigma[12] = 1.0/sigma_yIVDOSE_icg - 1.0*std::pow(-myIVDOSE_icg + yIVDOSE_icg, 2)/std::pow(sigma_yIVDOSE_icg, 3);
            break;
        case 13:
            dJydsigma[13] = 1.0/sigma_ycum_dose_icg - 1.0*std::pow(-mycum_dose_icg + ycum_dose_icg, 2)/std::pow(sigma_ycum_dose_icg, 3);
            break;
        case 14:
            dJydsigma[14] = 1.0/sigma_yFVre - 1.0*std::pow(-myFVre + yFVre, 2)/std::pow(sigma_yFVre, 3);
            break;
        case 15:
            dJydsigma[15] = 1.0/sigma_yFQre - 1.0*std::pow(-myFQre + yFQre, 2)/std::pow(sigma_yFQre, 3);
            break;
        case 16:
            dJydsigma[16] = 1.0/sigma_yBSA - 1.0*std::pow(-myBSA + yBSA, 2)/std::pow(sigma_yBSA, 3);
            break;
        case 17:
            dJydsigma[17] = 1.0/sigma_yCO - 1.0*std::pow(-myCO + yCO, 2)/std::pow(sigma_yCO, 3);
            break;
        case 18:
            dJydsigma[18] = 1.0/sigma_yQC - 1.0*std::pow(-myQC + yQC, 2)/std::pow(sigma_yQC, 3);
            break;
        case 19:
            dJydsigma[19] = 1.0/sigma_yQlu - 1.0*std::pow(-myQlu + yQlu, 2)/std::pow(sigma_yQlu, 3);
            break;
        case 20:
            dJydsigma[20] = 1.0/sigma_yQre - 1.0*std::pow(-myQre + yQre, 2)/std::pow(sigma_yQre, 3);
            break;
        case 21:
            dJydsigma[21] = 1.0/sigma_yQh - 1.0*std::pow(-myQh + yQh, 2)/std::pow(sigma_yQh, 3);
            break;
        case 22:
            dJydsigma[22] = 1.0/sigma_yQgi - 1.0*std::pow(-myQgi + yQgi, 2)/std::pow(sigma_yQgi, 3);
            break;
        case 23:
            dJydsigma[23] = 1.0/sigma_yQpo - 1.0*std::pow(-myQpo + yQpo, 2)/std::pow(sigma_yQpo, 3);
            break;
        case 24:
            dJydsigma[24] = 1.0/sigma_yQha - 1.0*std::pow(-myQha + yQha, 2)/std::pow(sigma_yQha, 3);
            break;
        case 25:
            dJydsigma[25] = 1.0/sigma_yKi_icg - 1.0*std::pow(-myKi_icg + yKi_icg, 2)/std::pow(sigma_yKi_icg, 3);
            break;
        case 26:
            dJydsigma[26] = 1.0/sigma_yAre_plasma_icg - 1.0*std::pow(-myAre_plasma_icg + yAre_plasma_icg, 2)/std::pow(sigma_yAre_plasma_icg, 3);
            break;
        case 27:
            dJydsigma[27] = 1.0/sigma_yXre_plasma_icg - 1.0*std::pow(-myXre_plasma_icg + yXre_plasma_icg, 2)/std::pow(sigma_yXre_plasma_icg, 3);
            break;
        case 28:
            dJydsigma[28] = 1.0/sigma_yMre_plasma_icg - 1.0*std::pow(-myMre_plasma_icg + yMre_plasma_icg, 2)/std::pow(sigma_yMre_plasma_icg, 3);
            break;
        case 29:
            dJydsigma[29] = 1.0/sigma_yAgi_plasma_icg - 1.0*std::pow(-myAgi_plasma_icg + yAgi_plasma_icg, 2)/std::pow(sigma_yAgi_plasma_icg, 3);
            break;
        case 30:
            dJydsigma[30] = 1.0/sigma_yXgi_plasma_icg - 1.0*std::pow(-myXgi_plasma_icg + yXgi_plasma_icg, 2)/std::pow(sigma_yXgi_plasma_icg, 3);
            break;
        case 31:
            dJydsigma[31] = 1.0/sigma_yMgi_plasma_icg - 1.0*std::pow(-myMgi_plasma_icg + yMgi_plasma_icg, 2)/std::pow(sigma_yMgi_plasma_icg, 3);
            break;
        case 32:
            dJydsigma[32] = 1.0/sigma_yAli_plasma_icg - 1.0*std::pow(-myAli_plasma_icg + yAli_plasma_icg, 2)/std::pow(sigma_yAli_plasma_icg, 3);
            break;
        case 33:
            dJydsigma[33] = 1.0/sigma_yXli_plasma_icg - 1.0*std::pow(-myXli_plasma_icg + yXli_plasma_icg, 2)/std::pow(sigma_yXli_plasma_icg, 3);
            break;
        case 34:
            dJydsigma[34] = 1.0/sigma_yMli_plasma_icg - 1.0*std::pow(-myMli_plasma_icg + yMli_plasma_icg, 2)/std::pow(sigma_yMli_plasma_icg, 3);
            break;
        case 35:
            dJydsigma[35] = 1.0/sigma_yAlu_plasma_icg - 1.0*std::pow(-myAlu_plasma_icg + yAlu_plasma_icg, 2)/std::pow(sigma_yAlu_plasma_icg, 3);
            break;
        case 36:
            dJydsigma[36] = 1.0/sigma_yXlu_plasma_icg - 1.0*std::pow(-myXlu_plasma_icg + yXlu_plasma_icg, 2)/std::pow(sigma_yXlu_plasma_icg, 3);
            break;
        case 37:
            dJydsigma[37] = 1.0/sigma_yMlu_plasma_icg - 1.0*std::pow(-myMlu_plasma_icg + yMlu_plasma_icg, 2)/std::pow(sigma_yMlu_plasma_icg, 3);
            break;
        case 38:
            dJydsigma[38] = 1.0/sigma_yAve_icg - 1.0*std::pow(-myAve_icg + yAve_icg, 2)/std::pow(sigma_yAve_icg, 3);
            break;
        case 39:
            dJydsigma[39] = 1.0/sigma_yXve_icg - 1.0*std::pow(-myXve_icg + yXve_icg, 2)/std::pow(sigma_yXve_icg, 3);
            break;
        case 40:
            dJydsigma[40] = 1.0/sigma_yMve_icg - 1.0*std::pow(-myMve_icg + yMve_icg, 2)/std::pow(sigma_yMve_icg, 3);
            break;
        case 41:
            dJydsigma[41] = 1.0/sigma_yAar_icg - 1.0*std::pow(-myAar_icg + yAar_icg, 2)/std::pow(sigma_yAar_icg, 3);
            break;
        case 42:
            dJydsigma[42] = 1.0/sigma_yXar_icg - 1.0*std::pow(-myXar_icg + yXar_icg, 2)/std::pow(sigma_yXar_icg, 3);
            break;
        case 43:
            dJydsigma[43] = 1.0/sigma_yMar_icg - 1.0*std::pow(-myMar_icg + yMar_icg, 2)/std::pow(sigma_yMar_icg, 3);
            break;
        case 44:
            dJydsigma[44] = 1.0/sigma_yApo_icg - 1.0*std::pow(-myApo_icg + yApo_icg, 2)/std::pow(sigma_yApo_icg, 3);
            break;
        case 45:
            dJydsigma[45] = 1.0/sigma_yXpo_icg - 1.0*std::pow(-myXpo_icg + yXpo_icg, 2)/std::pow(sigma_yXpo_icg, 3);
            break;
        case 46:
            dJydsigma[46] = 1.0/sigma_yMpo_icg - 1.0*std::pow(-myMpo_icg + yMpo_icg, 2)/std::pow(sigma_yMpo_icg, 3);
            break;
        case 47:
            dJydsigma[47] = 1.0/sigma_yAhv_icg - 1.0*std::pow(-myAhv_icg + yAhv_icg, 2)/std::pow(sigma_yAhv_icg, 3);
            break;
        case 48:
            dJydsigma[48] = 1.0/sigma_yXhv_icg - 1.0*std::pow(-myXhv_icg + yXhv_icg, 2)/std::pow(sigma_yXhv_icg, 3);
            break;
        case 49:
            dJydsigma[49] = 1.0/sigma_yMhv_icg - 1.0*std::pow(-myMhv_icg + yMhv_icg, 2)/std::pow(sigma_yMhv_icg, 3);
            break;
        case 50:
            dJydsigma[50] = 1.0/sigma_yER_icg - 1.0*std::pow(-myER_icg + yER_icg, 2)/std::pow(sigma_yER_icg, 3);
            break;
        case 51:
            dJydsigma[51] = 1.0/sigma_yCLinfusion_icg - 1.0*std::pow(-myCLinfusion_icg + yCLinfusion_icg, 2)/std::pow(sigma_yCLinfusion_icg, 3);
            break;
        case 52:
            dJydsigma[52] = 1.0/sigma_yVbi - 1.0*std::pow(-myVbi + yVbi, 2)/std::pow(sigma_yVbi, 3);
            break;
        case 53:
            dJydsigma[53] = 1.0/sigma_yVgi - 1.0*std::pow(-myVgi + yVgi, 2)/std::pow(sigma_yVgi, 3);
            break;
        case 54:
            dJydsigma[54] = 1.0/sigma_yVli - 1.0*std::pow(-myVli + yVli, 2)/std::pow(sigma_yVli, 3);
            break;
        case 55:
            dJydsigma[55] = 1.0/sigma_yVlu - 1.0*std::pow(-myVlu + yVlu, 2)/std::pow(sigma_yVlu, 3);
            break;
        case 56:
            dJydsigma[56] = 1.0/sigma_yVre - 1.0*std::pow(-myVre + yVre, 2)/std::pow(sigma_yVre, 3);
            break;
        case 57:
            dJydsigma[57] = 1.0/sigma_yVve - 1.0*std::pow(-myVve + yVve, 2)/std::pow(sigma_yVve, 3);
            break;
        case 58:
            dJydsigma[58] = 1.0/sigma_yVar - 1.0*std::pow(-myVar + yVar, 2)/std::pow(sigma_yVar, 3);
            break;
        case 59:
            dJydsigma[59] = 1.0/sigma_yVpo - 1.0*std::pow(-myVpo + yVpo, 2)/std::pow(sigma_yVpo, 3);
            break;
        case 60:
            dJydsigma[60] = 1.0/sigma_yVhv - 1.0*std::pow(-myVhv + yVhv, 2)/std::pow(sigma_yVhv, 3);
            break;
        case 61:
            dJydsigma[61] = 1.0/sigma_yVre_plasma - 1.0*std::pow(-myVre_plasma + yVre_plasma, 2)/std::pow(sigma_yVre_plasma, 3);
            break;
        case 62:
            dJydsigma[62] = 1.0/sigma_yVre_tissue - 1.0*std::pow(-myVre_tissue + yVre_tissue, 2)/std::pow(sigma_yVre_tissue, 3);
            break;
        case 63:
            dJydsigma[63] = 1.0/sigma_yVgi_plasma - 1.0*std::pow(-myVgi_plasma + yVgi_plasma, 2)/std::pow(sigma_yVgi_plasma, 3);
            break;
        case 64:
            dJydsigma[64] = 1.0/sigma_yVgi_tissue - 1.0*std::pow(-myVgi_tissue + yVgi_tissue, 2)/std::pow(sigma_yVgi_tissue, 3);
            break;
        case 65:
            dJydsigma[65] = 1.0/sigma_yVli_plasma - 1.0*std::pow(-myVli_plasma + yVli_plasma, 2)/std::pow(sigma_yVli_plasma, 3);
            break;
        case 66:
            dJydsigma[66] = 1.0/sigma_yVli_tissue - 1.0*std::pow(-myVli_tissue + yVli_tissue, 2)/std::pow(sigma_yVli_tissue, 3);
            break;
        case 67:
            dJydsigma[67] = 1.0/sigma_yVlu_plasma - 1.0*std::pow(-myVlu_plasma + yVlu_plasma, 2)/std::pow(sigma_yVlu_plasma, 3);
            break;
        case 68:
            dJydsigma[68] = 1.0/sigma_yVlu_tissue - 1.0*std::pow(-myVlu_tissue + yVlu_tissue, 2)/std::pow(sigma_yVlu_tissue, 3);
            break;
        case 69:
            dJydsigma[69] = 1.0/sigma_yVfeces - 1.0*std::pow(-myVfeces + yVfeces, 2)/std::pow(sigma_yVfeces, 3);
            break;
    }
}

} // namespace model_icg_sd
} // namespace amici
