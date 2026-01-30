#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "p.h"
#include "w.h"
#include "tcl.h"
#include "dtcldp.h"

namespace amici {
namespace model_icg_sd {

void dydp_icg_sd(realtype *dydp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const realtype *w, const realtype *tcl, const realtype *dtcldp){
    switch(ip) {
        case 2:
            dydp[15] = COBW*f_cardiac_output*(-3.0/50.0*FQh*f_bloodflow*f_exercise/Qlu + (3.0/50.0)*FQlu*Qh/std::pow(Qlu, 2));
            dydp[16] = 0.013049716999999997*std::pow(BW, -0.46220000000000006)*std::pow(HEIGHT, 0.39639999999999997);
            dydp[17] = COBW*f_cardiac_output;
            dydp[18] = (3.0/50.0)*COBW*f_cardiac_output;
            dydp[19] = (3.0/50.0)*COBW*FQlu*f_cardiac_output;
            dydp[20] = COBW*f_cardiac_output*(-3.0/50.0*FQh*QC*f_bloodflow*f_exercise/Qlu + (3.0/50.0)*FQlu*QC*Qh/std::pow(Qlu, 2) + (3.0/50.0)*FQre);
            dydp[21] = (3.0/50.0)*COBW*FQh*f_bloodflow*f_cardiac_output*f_exercise;
            dydp[22] = (3.0/50.0)*COBW*FQgi*f_bloodflow*f_cardiac_output;
            dydp[23] = (3.0/50.0)*COBW*FQgi*f_bloodflow*f_cardiac_output;
            dydp[24] = COBW*f_cardiac_output*(-3.0/50.0*FQgi*f_bloodflow + (3.0/50.0)*FQh*f_bloodflow*f_exercise);
            dydp[26] = Cre_plasma_icg*FVre*Fblood*(1 - HCT);
            dydp[27] = Cre_plasma_icg*FVre*Fblood*Mr_icg*(1 - HCT);
            dydp[28] = FVre*(-Are_plasma_icg*Mr_icg/(Fblood*std::pow(Vre, 2)*(1 - HCT)) + Cre_plasma_icg*Mr_icg/Vre);
            dydp[29] = Cgi_plasma_icg*FVgi*Fblood*(1 - HCT);
            dydp[30] = Cgi_plasma_icg*FVgi*Fblood*Mr_icg*(1 - HCT);
            dydp[31] = FVgi*(-Agi_plasma_icg*Mr_icg/(Fblood*std::pow(Vgi, 2)*(1 - HCT)) + Cgi_plasma_icg*Mr_icg/Vgi);
            dydp[32] = Cli_plasma_icg*FVli*Fblood*(1 - HCT)*(1 - resection_rate);
            dydp[33] = Cli_plasma_icg*FVli*Fblood*Mr_icg*(1 - HCT)*(1 - resection_rate);
            dydp[34] = FVli*(1 - resection_rate)*(-Ali_plasma_icg*Mr_icg/(Fblood*std::pow(Vli, 2)*(1 - HCT)) + Cli_plasma_icg*Mr_icg/Vli);
            dydp[35] = Clu_plasma_icg*FVlu*Fblood*(1 - HCT);
            dydp[36] = Clu_plasma_icg*FVlu*Fblood*Mr_icg*(1 - HCT);
            dydp[37] = FVlu*(-Alu_plasma_icg*Mr_icg/(Fblood*std::pow(Vlu, 2)*(1 - HCT)) + Clu_plasma_icg*Mr_icg/Vlu);
            dydp[38] = Cve_icg*(1 - HCT)*(-FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVve);
            dydp[39] = Cve_icg*Mr_icg*(1 - HCT)*(-FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVve);
            dydp[40] = -Ave_icg*Mr_icg*(-FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVve)/((1 - HCT)*std::pow(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve, 2)) + Cve_icg*Mr_icg*(-FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVve)/(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);
            dydp[41] = Car_icg*(1 - HCT)*(-FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVar);
            dydp[42] = Car_icg*Mr_icg*(1 - HCT)*(-FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVar);
            dydp[43] = -Aar_icg*Mr_icg*(-FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVar)/((1 - HCT)*std::pow(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar, 2)) + Car_icg*Mr_icg*(-FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVar)/(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);
            dydp[44] = Cpo_icg*(1 - HCT)*(-FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVpo);
            dydp[45] = Cpo_icg*Mr_icg*(1 - HCT)*(-FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVpo);
            dydp[46] = -Apo_icg*Mr_icg*(-FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVpo)/((1 - HCT)*std::pow(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo, 2)) + Cpo_icg*Mr_icg*(-FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVpo)/(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);
            dydp[47] = Chv_icg*(1 - HCT)*(-FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVhv);
            dydp[48] = Chv_icg*Mr_icg*(1 - HCT)*(-FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVhv);
            dydp[49] = -Ahv_icg*Mr_icg*(-FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVhv)/((1 - HCT)*std::pow(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv, 2)) + Chv_icg*Mr_icg*(-FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVhv)/(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);
            dydp[52] = FVbi;
            dydp[53] = FVgi;
            dydp[54] = FVli*(1 - resection_rate);
            dydp[55] = FVlu;
            dydp[56] = FVre;
            dydp[57] = (1 - HCT)*(-FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVve);
            dydp[58] = (1 - HCT)*(-FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVar);
            dydp[59] = (1 - HCT)*(-FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVpo);
            dydp[60] = (1 - HCT)*(-FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + FVhv);
            dydp[61] = FVre*Fblood*(1 - HCT);
            dydp[62] = FVre*(1 - Fblood);
            dydp[63] = FVgi*Fblood*(1 - HCT);
            dydp[64] = FVgi*(1 - Fblood);
            dydp[65] = FVli*Fblood*(1 - HCT)*(1 - resection_rate);
            dydp[66] = FVli*(1 - Fblood)*(1 - f_tissue_loss)*(1 - resection_rate);
            dydp[67] = FVlu*Fblood*(1 - HCT);
            dydp[68] = FVlu*(1 - Fblood);
            break;
        case 3:
            dydp[16] = 0.0096186459999999981*std::pow(BW, 0.53779999999999994)*std::pow(HEIGHT, -0.60360000000000003);
            break;
        case 4:
            dydp[15] = BW*f_cardiac_output*(-3.0/50.0*FQh*f_bloodflow*f_exercise/Qlu + (3.0/50.0)*FQlu*Qh/std::pow(Qlu, 2));
            dydp[17] = BW*f_cardiac_output;
            dydp[18] = (3.0/50.0)*BW*f_cardiac_output;
            dydp[19] = (3.0/50.0)*BW*FQlu*f_cardiac_output;
            dydp[20] = BW*f_cardiac_output*(-3.0/50.0*FQh*QC*f_bloodflow*f_exercise/Qlu + (3.0/50.0)*FQlu*QC*Qh/std::pow(Qlu, 2) + (3.0/50.0)*FQre);
            dydp[21] = (3.0/50.0)*BW*FQh*f_bloodflow*f_cardiac_output*f_exercise;
            dydp[22] = (3.0/50.0)*BW*FQgi*f_bloodflow*f_cardiac_output;
            dydp[23] = (3.0/50.0)*BW*FQgi*f_bloodflow*f_cardiac_output;
            dydp[24] = BW*f_cardiac_output*(-3.0/50.0*FQgi*f_bloodflow + (3.0/50.0)*FQh*f_bloodflow*f_exercise);
            break;
        case 5:
            dydp[26] = Cre_plasma_icg*Vre*(1 - HCT);
            dydp[27] = Cre_plasma_icg*Mr_icg*Vre*(1 - HCT);
            dydp[28] = -Are_plasma_icg*Mr_icg/(std::pow(Fblood, 2)*Vre*(1 - HCT)) + Cre_plasma_icg*Mr_icg/Fblood;
            dydp[29] = Cgi_plasma_icg*Vgi*(1 - HCT);
            dydp[30] = Cgi_plasma_icg*Mr_icg*Vgi*(1 - HCT);
            dydp[31] = -Agi_plasma_icg*Mr_icg/(std::pow(Fblood, 2)*Vgi*(1 - HCT)) + Cgi_plasma_icg*Mr_icg/Fblood;
            dydp[32] = Cli_plasma_icg*Vli*(1 - HCT);
            dydp[33] = Cli_plasma_icg*Mr_icg*Vli*(1 - HCT);
            dydp[34] = -Ali_plasma_icg*Mr_icg/(std::pow(Fblood, 2)*Vli*(1 - HCT)) + Cli_plasma_icg*Mr_icg/Fblood;
            dydp[35] = Clu_plasma_icg*Vlu*(1 - HCT);
            dydp[36] = Clu_plasma_icg*Mr_icg*Vlu*(1 - HCT);
            dydp[37] = -Alu_plasma_icg*Mr_icg/(std::pow(Fblood, 2)*Vlu*(1 - HCT)) + Clu_plasma_icg*Mr_icg/Fblood;
            dydp[38] = -BW*Cve_icg*FVve*(1 - HCT)*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve);
            dydp[39] = -BW*Cve_icg*FVve*Mr_icg*(1 - HCT)*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve);
            dydp[40] = Ave_icg*BW*FVve*Mr_icg*(-FVar - FVhv - FVpo - FVve + 1)/((1 - HCT)*std::pow(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve, 2)*(FVar + FVhv + FVpo + FVve)) - BW*Cve_icg*FVve*Mr_icg*(-FVar - FVhv - FVpo - FVve + 1)/((-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve)*(FVar + FVhv + FVpo + FVve));
            dydp[41] = -BW*Car_icg*FVar*(1 - HCT)*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve);
            dydp[42] = -BW*Car_icg*FVar*Mr_icg*(1 - HCT)*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve);
            dydp[43] = Aar_icg*BW*FVar*Mr_icg*(-FVar - FVhv - FVpo - FVve + 1)/((1 - HCT)*std::pow(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar, 2)*(FVar + FVhv + FVpo + FVve)) - BW*Car_icg*FVar*Mr_icg*(-FVar - FVhv - FVpo - FVve + 1)/((-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar)*(FVar + FVhv + FVpo + FVve));
            dydp[44] = -BW*Cpo_icg*FVpo*(1 - HCT)*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve);
            dydp[45] = -BW*Cpo_icg*FVpo*Mr_icg*(1 - HCT)*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve);
            dydp[46] = Apo_icg*BW*FVpo*Mr_icg*(-FVar - FVhv - FVpo - FVve + 1)/((1 - HCT)*std::pow(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo, 2)*(FVar + FVhv + FVpo + FVve)) - BW*Cpo_icg*FVpo*Mr_icg*(-FVar - FVhv - FVpo - FVve + 1)/((-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo)*(FVar + FVhv + FVpo + FVve));
            dydp[47] = -BW*Chv_icg*FVhv*(1 - HCT)*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve);
            dydp[48] = -BW*Chv_icg*FVhv*Mr_icg*(1 - HCT)*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve);
            dydp[49] = Ahv_icg*BW*FVhv*Mr_icg*(-FVar - FVhv - FVpo - FVve + 1)/((1 - HCT)*std::pow(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv, 2)*(FVar + FVhv + FVpo + FVve)) - BW*Chv_icg*FVhv*Mr_icg*(-FVar - FVhv - FVpo - FVve + 1)/((-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv)*(FVar + FVhv + FVpo + FVve));
            dydp[57] = -BW*FVve*(1 - HCT)*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve);
            dydp[58] = -BW*FVar*(1 - HCT)*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve);
            dydp[59] = -BW*FVpo*(1 - HCT)*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve);
            dydp[60] = -BW*FVhv*(1 - HCT)*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve);
            dydp[61] = BW*FVre*(1 - HCT);
            dydp[62] = -BW*FVre;
            dydp[63] = BW*FVgi*(1 - HCT);
            dydp[64] = -BW*FVgi;
            dydp[65] = BW*FVli*(1 - HCT)*(1 - resection_rate);
            dydp[66] = -BW*FVli*(1 - f_tissue_loss)*(1 - resection_rate);
            dydp[67] = BW*FVlu*(1 - HCT);
            dydp[68] = -BW*FVlu;
            break;
        case 6:
            dydp[26] = -Cre_plasma_icg*Fblood*Vre;
            dydp[27] = -Cre_plasma_icg*Fblood*Mr_icg*Vre;
            dydp[28] = Are_plasma_icg*Mr_icg/(Fblood*Vre*std::pow(1 - HCT, 2)) - Cre_plasma_icg*Mr_icg/(1 - HCT);
            dydp[29] = -Cgi_plasma_icg*Fblood*Vgi;
            dydp[30] = -Cgi_plasma_icg*Fblood*Mr_icg*Vgi;
            dydp[31] = Agi_plasma_icg*Mr_icg/(Fblood*Vgi*std::pow(1 - HCT, 2)) - Cgi_plasma_icg*Mr_icg/(1 - HCT);
            dydp[32] = -Cli_plasma_icg*Fblood*Vli;
            dydp[33] = -Cli_plasma_icg*Fblood*Mr_icg*Vli;
            dydp[34] = Ali_plasma_icg*Mr_icg/(Fblood*Vli*std::pow(1 - HCT, 2)) - Cli_plasma_icg*Mr_icg/(1 - HCT);
            dydp[35] = -Clu_plasma_icg*Fblood*Vlu;
            dydp[36] = -Clu_plasma_icg*Fblood*Mr_icg*Vlu;
            dydp[37] = Alu_plasma_icg*Mr_icg/(Fblood*Vlu*std::pow(1 - HCT, 2)) - Clu_plasma_icg*Mr_icg/(1 - HCT);
            dydp[38] = -Cve_icg*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);
            dydp[39] = -Cve_icg*Mr_icg*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);
            dydp[40] = Ave_icg*Mr_icg/(std::pow(1 - HCT, 2)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve)) - Cve_icg*Mr_icg/(1 - HCT);
            dydp[41] = -Car_icg*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);
            dydp[42] = -Car_icg*Mr_icg*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);
            dydp[43] = Aar_icg*Mr_icg/(std::pow(1 - HCT, 2)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar)) - Car_icg*Mr_icg/(1 - HCT);
            dydp[44] = -Cpo_icg*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);
            dydp[45] = -Cpo_icg*Mr_icg*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);
            dydp[46] = Apo_icg*Mr_icg/(std::pow(1 - HCT, 2)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo)) - Cpo_icg*Mr_icg/(1 - HCT);
            dydp[47] = -Chv_icg*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);
            dydp[48] = -Chv_icg*Mr_icg*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);
            dydp[49] = Ahv_icg*Mr_icg/(std::pow(1 - HCT, 2)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv)) - Chv_icg*Mr_icg/(1 - HCT);
            dydp[57] = BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) - BW*FVve;
            dydp[58] = BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) - BW*FVar;
            dydp[59] = BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) - BW*FVpo;
            dydp[60] = BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) - BW*FVhv;
            dydp[61] = -BW*FVre*Fblood;
            dydp[63] = -BW*FVgi*Fblood;
            dydp[65] = -BW*FVli*Fblood*(1 - resection_rate);
            dydp[67] = -BW*FVlu*Fblood;
            break;
        case 7:
            dydp[14] = -1;
            dydp[26] = -BW*Cre_plasma_icg*Fblood*(1 - HCT);
            dydp[27] = -BW*Cre_plasma_icg*Fblood*Mr_icg*(1 - HCT);
            dydp[28] = Are_plasma_icg*BW*Mr_icg/(Fblood*std::pow(Vre, 2)*(1 - HCT)) - BW*Cre_plasma_icg*Mr_icg/Vre;
            dydp[29] = BW*Cgi_plasma_icg*Fblood*(1 - HCT);
            dydp[30] = BW*Cgi_plasma_icg*Fblood*Mr_icg*(1 - HCT);
            dydp[31] = BW*(-Agi_plasma_icg*Mr_icg/(Fblood*std::pow(Vgi, 2)*(1 - HCT)) + Cgi_plasma_icg*Mr_icg/Vgi);
            dydp[53] = BW;
            dydp[56] = -BW;
            dydp[61] = -BW*Fblood*(1 - HCT);
            dydp[62] = -BW*(1 - Fblood);
            dydp[63] = BW*Fblood*(1 - HCT);
            dydp[64] = BW*(1 - Fblood);
            break;
        case 8:
            dydp[14] = -1;
            dydp[26] = -BW*Cre_plasma_icg*Fblood*(1 - HCT);
            dydp[27] = -BW*Cre_plasma_icg*Fblood*Mr_icg*(1 - HCT);
            dydp[28] = Are_plasma_icg*BW*Mr_icg/(Fblood*std::pow(Vre, 2)*(1 - HCT)) - BW*Cre_plasma_icg*Mr_icg/Vre;
            dydp[52] = BW;
            dydp[56] = -BW;
            dydp[61] = -BW*Fblood*(1 - HCT);
            dydp[62] = -BW*(1 - Fblood);
            break;
        case 9:
            dydp[14] = -1;
            dydp[26] = -BW*Cre_plasma_icg*Fblood*(1 - HCT);
            dydp[27] = -BW*Cre_plasma_icg*Fblood*Mr_icg*(1 - HCT);
            dydp[28] = Are_plasma_icg*BW*Mr_icg/(Fblood*std::pow(Vre, 2)*(1 - HCT)) - BW*Cre_plasma_icg*Mr_icg/Vre;
            dydp[32] = BW*Cli_plasma_icg*Fblood*(1 - HCT)*(1 - resection_rate);
            dydp[33] = BW*Cli_plasma_icg*Fblood*Mr_icg*(1 - HCT)*(1 - resection_rate);
            dydp[34] = BW*(1 - resection_rate)*(-Ali_plasma_icg*Mr_icg/(Fblood*std::pow(Vli, 2)*(1 - HCT)) + Cli_plasma_icg*Mr_icg/Vli);
            dydp[54] = BW*(1 - resection_rate);
            dydp[56] = -BW;
            dydp[61] = -BW*Fblood*(1 - HCT);
            dydp[62] = -BW*(1 - Fblood);
            dydp[65] = BW*Fblood*(1 - HCT)*(1 - resection_rate);
            dydp[66] = BW*(1 - Fblood)*(1 - f_tissue_loss)*(1 - resection_rate);
            break;
        case 10:
            dydp[14] = -1;
            dydp[26] = -BW*Cre_plasma_icg*Fblood*(1 - HCT);
            dydp[27] = -BW*Cre_plasma_icg*Fblood*Mr_icg*(1 - HCT);
            dydp[28] = Are_plasma_icg*BW*Mr_icg/(Fblood*std::pow(Vre, 2)*(1 - HCT)) - BW*Cre_plasma_icg*Mr_icg/Vre;
            dydp[35] = BW*Clu_plasma_icg*Fblood*(1 - HCT);
            dydp[36] = BW*Clu_plasma_icg*Fblood*Mr_icg*(1 - HCT);
            dydp[37] = BW*(-Alu_plasma_icg*Mr_icg/(Fblood*std::pow(Vlu, 2)*(1 - HCT)) + Clu_plasma_icg*Mr_icg/Vlu);
            dydp[55] = BW;
            dydp[56] = -BW;
            dydp[61] = -BW*Fblood*(1 - HCT);
            dydp[62] = -BW*(1 - Fblood);
            dydp[67] = BW*Fblood*(1 - HCT);
            dydp[68] = BW*(1 - Fblood);
            break;
        case 11:
            dydp[14] = -1;
            dydp[26] = -BW*Cre_plasma_icg*Fblood*(1 - HCT);
            dydp[27] = -BW*Cre_plasma_icg*Fblood*Mr_icg*(1 - HCT);
            dydp[28] = Are_plasma_icg*BW*Mr_icg/(Fblood*std::pow(Vre, 2)*(1 - HCT)) - BW*Cre_plasma_icg*Mr_icg/Vre;
            dydp[38] = Cve_icg*(1 - HCT)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW);
            dydp[39] = Cve_icg*Mr_icg*(1 - HCT)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW);
            dydp[40] = -Ave_icg*Mr_icg*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW)/((1 - HCT)*std::pow(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve, 2)) + Cve_icg*Mr_icg*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW)/(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);
            dydp[41] = Car_icg*(1 - HCT)*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[42] = Car_icg*Mr_icg*(1 - HCT)*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[43] = -Aar_icg*Mr_icg*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/((1 - HCT)*std::pow(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar, 2)) + Car_icg*Mr_icg*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);
            dydp[44] = Cpo_icg*(1 - HCT)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[45] = Cpo_icg*Mr_icg*(1 - HCT)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[46] = -Apo_icg*Mr_icg*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/((1 - HCT)*std::pow(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo, 2)) + Cpo_icg*Mr_icg*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);
            dydp[47] = Chv_icg*(1 - HCT)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[48] = Chv_icg*Mr_icg*(1 - HCT)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[49] = -Ahv_icg*Mr_icg*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/((1 - HCT)*std::pow(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv, 2)) + Chv_icg*Mr_icg*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);
            dydp[56] = -BW;
            dydp[57] = (1 - HCT)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW);
            dydp[58] = (1 - HCT)*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[59] = (1 - HCT)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[60] = (1 - HCT)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[61] = -BW*Fblood*(1 - HCT);
            dydp[62] = -BW*(1 - Fblood);
            break;
        case 12:
            dydp[14] = -1;
            dydp[26] = -BW*Cre_plasma_icg*Fblood*(1 - HCT);
            dydp[27] = -BW*Cre_plasma_icg*Fblood*Mr_icg*(1 - HCT);
            dydp[28] = Are_plasma_icg*BW*Mr_icg/(Fblood*std::pow(Vre, 2)*(1 - HCT)) - BW*Cre_plasma_icg*Mr_icg/Vre;
            dydp[38] = Cve_icg*(1 - HCT)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[39] = Cve_icg*Mr_icg*(1 - HCT)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[40] = -Ave_icg*Mr_icg*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/((1 - HCT)*std::pow(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve, 2)) + Cve_icg*Mr_icg*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);
            dydp[41] = Car_icg*(1 - HCT)*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW);
            dydp[42] = Car_icg*Mr_icg*(1 - HCT)*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW);
            dydp[43] = -Aar_icg*Mr_icg*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW)/((1 - HCT)*std::pow(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar, 2)) + Car_icg*Mr_icg*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW)/(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);
            dydp[44] = Cpo_icg*(1 - HCT)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[45] = Cpo_icg*Mr_icg*(1 - HCT)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[46] = -Apo_icg*Mr_icg*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/((1 - HCT)*std::pow(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo, 2)) + Cpo_icg*Mr_icg*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);
            dydp[47] = Chv_icg*(1 - HCT)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[48] = Chv_icg*Mr_icg*(1 - HCT)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[49] = -Ahv_icg*Mr_icg*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/((1 - HCT)*std::pow(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv, 2)) + Chv_icg*Mr_icg*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);
            dydp[56] = -BW;
            dydp[57] = (1 - HCT)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[58] = (1 - HCT)*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW);
            dydp[59] = (1 - HCT)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[60] = (1 - HCT)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[61] = -BW*Fblood*(1 - HCT);
            dydp[62] = -BW*(1 - Fblood);
            break;
        case 13:
            dydp[14] = -1;
            dydp[26] = -BW*Cre_plasma_icg*Fblood*(1 - HCT);
            dydp[27] = -BW*Cre_plasma_icg*Fblood*Mr_icg*(1 - HCT);
            dydp[28] = Are_plasma_icg*BW*Mr_icg/(Fblood*std::pow(Vre, 2)*(1 - HCT)) - BW*Cre_plasma_icg*Mr_icg/Vre;
            dydp[38] = Cve_icg*(1 - HCT)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[39] = Cve_icg*Mr_icg*(1 - HCT)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[40] = -Ave_icg*Mr_icg*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/((1 - HCT)*std::pow(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve, 2)) + Cve_icg*Mr_icg*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);
            dydp[41] = Car_icg*(1 - HCT)*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[42] = Car_icg*Mr_icg*(1 - HCT)*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[43] = -Aar_icg*Mr_icg*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/((1 - HCT)*std::pow(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar, 2)) + Car_icg*Mr_icg*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);
            dydp[44] = Cpo_icg*(1 - HCT)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW);
            dydp[45] = Cpo_icg*Mr_icg*(1 - HCT)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW);
            dydp[46] = -Apo_icg*Mr_icg*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW)/((1 - HCT)*std::pow(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo, 2)) + Cpo_icg*Mr_icg*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW)/(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);
            dydp[47] = Chv_icg*(1 - HCT)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[48] = Chv_icg*Mr_icg*(1 - HCT)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[49] = -Ahv_icg*Mr_icg*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/((1 - HCT)*std::pow(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv, 2)) + Chv_icg*Mr_icg*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);
            dydp[56] = -BW;
            dydp[57] = (1 - HCT)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[58] = (1 - HCT)*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[59] = (1 - HCT)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW);
            dydp[60] = (1 - HCT)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[61] = -BW*Fblood*(1 - HCT);
            dydp[62] = -BW*(1 - Fblood);
            break;
        case 14:
            dydp[14] = -1;
            dydp[26] = -BW*Cre_plasma_icg*Fblood*(1 - HCT);
            dydp[27] = -BW*Cre_plasma_icg*Fblood*Mr_icg*(1 - HCT);
            dydp[28] = Are_plasma_icg*BW*Mr_icg/(Fblood*std::pow(Vre, 2)*(1 - HCT)) - BW*Cre_plasma_icg*Mr_icg/Vre;
            dydp[38] = Cve_icg*(1 - HCT)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[39] = Cve_icg*Mr_icg*(1 - HCT)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[40] = -Ave_icg*Mr_icg*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/((1 - HCT)*std::pow(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve, 2)) + Cve_icg*Mr_icg*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);
            dydp[41] = Car_icg*(1 - HCT)*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[42] = Car_icg*Mr_icg*(1 - HCT)*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[43] = -Aar_icg*Mr_icg*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/((1 - HCT)*std::pow(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar, 2)) + Car_icg*Mr_icg*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);
            dydp[44] = Cpo_icg*(1 - HCT)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[45] = Cpo_icg*Mr_icg*(1 - HCT)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[46] = -Apo_icg*Mr_icg*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/((1 - HCT)*std::pow(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo, 2)) + Cpo_icg*Mr_icg*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2))/(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);
            dydp[47] = Chv_icg*(1 - HCT)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW);
            dydp[48] = Chv_icg*Mr_icg*(1 - HCT)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW);
            dydp[49] = -Ahv_icg*Mr_icg*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW)/((1 - HCT)*std::pow(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv, 2)) + Chv_icg*Mr_icg*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW)/(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);
            dydp[56] = -BW;
            dydp[57] = (1 - HCT)*(BW*FVve*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[58] = (1 - HCT)*(BW*FVar*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[59] = (1 - HCT)*(BW*FVpo*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2));
            dydp[60] = (1 - HCT)*(BW*FVhv*Fblood/(FVar + FVhv + FVpo + FVve) + BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/std::pow(FVar + FVhv + FVpo + FVve, 2) - BW*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW);
            dydp[61] = -BW*Fblood*(1 - HCT);
            dydp[62] = -BW*(1 - Fblood);
            break;
        case 15:
            dydp[22] = QC*f_bloodflow;
            dydp[23] = QC*f_bloodflow;
            dydp[24] = -QC*f_bloodflow;
            break;
        case 16:
            dydp[15] = -QC*f_bloodflow*f_exercise/Qlu;
            dydp[20] = -std::pow(QC, 2)*f_bloodflow*f_exercise/Qlu;
            dydp[21] = QC*f_bloodflow*f_exercise;
            dydp[24] = QC*f_bloodflow*f_exercise;
            break;
        case 17:
            dydp[15] = QC*Qh/std::pow(Qlu, 2);
            dydp[19] = QC;
            dydp[20] = std::pow(QC, 2)*Qh/std::pow(Qlu, 2);
            break;
        case 20:
            dydp[66] = -BW*FVli*(1 - Fblood)*(1 - resection_rate);
            break;
        case 21:
            dydp[15] = -FQh*QC*f_exercise/Qlu;
            dydp[20] = -FQh*std::pow(QC, 2)*f_exercise/Qlu;
            dydp[21] = FQh*QC*f_exercise;
            dydp[22] = FQgi*QC;
            dydp[23] = FQgi*QC;
            dydp[24] = -FQgi*QC + FQh*QC*f_exercise;
            break;
        case 22:
            dydp[15] = BW*COBW*(-3.0/50.0*FQh*f_bloodflow*f_exercise/Qlu + (3.0/50.0)*FQlu*Qh/std::pow(Qlu, 2));
            dydp[17] = BW*COBW;
            dydp[18] = (3.0/50.0)*BW*COBW;
            dydp[19] = (3.0/50.0)*BW*COBW*FQlu;
            dydp[20] = BW*COBW*(-3.0/50.0*FQh*QC*f_bloodflow*f_exercise/Qlu + (3.0/50.0)*FQlu*QC*Qh/std::pow(Qlu, 2) + (3.0/50.0)*FQre);
            dydp[21] = (3.0/50.0)*BW*COBW*FQh*f_bloodflow*f_exercise;
            dydp[22] = (3.0/50.0)*BW*COBW*FQgi*f_bloodflow;
            dydp[23] = (3.0/50.0)*BW*COBW*FQgi*f_bloodflow;
            dydp[24] = BW*COBW*(-3.0/50.0*FQgi*f_bloodflow + (3.0/50.0)*FQh*f_bloodflow*f_exercise);
            break;
        case 23:
            dydp[15] = -FQh*QC*f_bloodflow/Qlu;
            dydp[20] = -FQh*std::pow(QC, 2)*f_bloodflow/Qlu;
            dydp[21] = FQh*QC*f_bloodflow;
            dydp[24] = FQh*QC*f_bloodflow;
            break;
        case 24:
            dydp[32] = -BW*Cli_plasma_icg*FVli*Fblood*(1 - HCT);
            dydp[33] = -BW*Cli_plasma_icg*FVli*Fblood*Mr_icg*(1 - HCT);
            dydp[34] = -BW*FVli*(-Ali_plasma_icg*Mr_icg/(Fblood*std::pow(Vli, 2)*(1 - HCT)) + Cli_plasma_icg*Mr_icg/Vli);
            dydp[54] = -BW*FVli;
            dydp[65] = -BW*FVli*Fblood*(1 - HCT);
            dydp[66] = -BW*FVli*(1 - Fblood)*(1 - f_tissue_loss);
            break;
        case 25:
            dydp[27] = Are_plasma_icg;
            dydp[28] = Are_plasma_icg/(Fblood*Vre*(1 - HCT));
            dydp[30] = Agi_plasma_icg;
            dydp[31] = Agi_plasma_icg/(Fblood*Vgi*(1 - HCT));
            dydp[33] = Ali_plasma_icg;
            dydp[34] = Ali_plasma_icg/(Fblood*Vli*(1 - HCT));
            dydp[36] = Alu_plasma_icg;
            dydp[37] = Alu_plasma_icg/(Fblood*Vlu*(1 - HCT));
            dydp[39] = Ave_icg;
            dydp[40] = Ave_icg/((1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve));
            dydp[42] = Aar_icg;
            dydp[43] = Aar_icg/((1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar));
            dydp[45] = Apo_icg;
            dydp[46] = Apo_icg/((1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo));
            dydp[48] = Ahv_icg;
            dydp[49] = Ahv_icg/((1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv));
            dydp[51] = -Ri_icg/(std::pow(Mr_icg, 2)*(Cve_icg + 9.9999999999999998e-13));
            break;
        case 26:
            dydp[25] = -41.579999999999998/std::pow(ti_icg, 2);
            break;
        case 27:
            dydp[51] = 1/(Mr_icg*(Cve_icg + 9.9999999999999998e-13));
            break;
    }
}

} // namespace model_icg_sd
} // namespace amici
