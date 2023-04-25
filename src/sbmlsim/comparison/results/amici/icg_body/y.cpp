#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "p.h"
#include "w.h"

namespace amici {
namespace model_icg_body {

void y_icg_body(realtype *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w){
    y[0] = Cre_plasma_icg;
    y[1] = Cgi_plasma_icg;
    y[2] = Cli_plasma_icg;
    y[3] = Clu_plasma_icg;
    y[4] = Cve_icg;
    y[5] = Car_icg;
    y[6] = Cpo_icg;
    y[7] = Chv_icg;
    y[8] = Afeces_icg;
    y[9] = LI__bil_ext;
    y[10] = LI__icg;
    y[11] = LI__icg_bi;
    y[12] = IVDOSE_icg;
    y[13] = cum_dose_icg;
    y[14] = -FVar - FVbi - FVgi - FVhv - FVli - FVlu - FVpo - FVve + 1;
    y[15] = -Qh/Qlu + 1;
    y[16] = 0.024264999999999998*std::pow(BW, 0.53779999999999994)*std::pow(HEIGHT, 0.39639999999999997);
    y[17] = BW*COBW*f_cardiac_output;
    y[18] = (3.0/50.0)*CO;
    y[19] = FQlu*QC;
    y[20] = FQre*QC;
    y[21] = FQh*QC*f_bloodflow*f_exercise;
    y[22] = FQgi*QC*f_bloodflow;
    y[23] = Qgi;
    y[24] = Qh - Qpo;
    y[25] = 41.579999999999998/ti_icg;
    y[26] = Cre_plasma_icg*Fblood*Vre*(1 - HCT);
    y[27] = Are_plasma_icg*Mr_icg;
    y[28] = Are_plasma_icg*Mr_icg/(Fblood*Vre*(1 - HCT));
    y[29] = Cgi_plasma_icg*Fblood*Vgi*(1 - HCT);
    y[30] = Agi_plasma_icg*Mr_icg;
    y[31] = Agi_plasma_icg*Mr_icg/(Fblood*Vgi*(1 - HCT));
    y[32] = Cli_plasma_icg*Fblood*Vli*(1 - HCT);
    y[33] = Ali_plasma_icg*Mr_icg;
    y[34] = Ali_plasma_icg*Mr_icg/(Fblood*Vli*(1 - HCT));
    y[35] = Clu_plasma_icg*Fblood*Vlu*(1 - HCT);
    y[36] = Alu_plasma_icg*Mr_icg;
    y[37] = Alu_plasma_icg*Mr_icg/(Fblood*Vlu*(1 - HCT));
    y[38] = Cve_icg*(1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);
    y[39] = Ave_icg*Mr_icg;
    y[40] = Ave_icg*Mr_icg/((1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve));
    y[41] = Car_icg*(1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);
    y[42] = Aar_icg*Mr_icg;
    y[43] = Aar_icg*Mr_icg/((1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar));
    y[44] = Cpo_icg*(1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);
    y[45] = Apo_icg*Mr_icg;
    y[46] = Apo_icg*Mr_icg/((1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo));
    y[47] = Chv_icg*(1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);
    y[48] = Ahv_icg*Mr_icg;
    y[49] = Ahv_icg*Mr_icg/((1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv));
    y[50] = (Car_icg - Chv_icg)/(Car_icg + 9.9999999999999995e-8);
    y[51] = Ri_icg/(Mr_icg*(Cve_icg + 9.9999999999999998e-13));
    y[52] = BW*FVbi;
    y[53] = BW*FVgi;
    y[54] = BW*FVli*(1 - resection_rate);
    y[55] = BW*FVlu;
    y[56] = BW*FVre;
    y[57] = (1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);
    y[58] = (1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);
    y[59] = (1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);
    y[60] = (1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);
    y[61] = BW*FVre*Fblood*(1 - HCT);
    y[62] = BW*FVre*(1 - Fblood);
    y[63] = BW*FVgi*Fblood*(1 - HCT);
    y[64] = BW*FVgi*(1 - Fblood);
    y[65] = BW*FVli*Fblood*(1 - HCT)*(1 - resection_rate);
    y[66] = BW*FVli*(1 - Fblood)*(1 - f_tissue_loss)*(1 - resection_rate);
    y[67] = BW*FVlu*Fblood*(1 - HCT);
    y[68] = BW*FVlu*(1 - Fblood);
    y[69] = 1.0;
}

} // namespace model_icg_body
} // namespace amici
