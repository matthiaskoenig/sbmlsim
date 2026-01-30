#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "p.h"
#include "w.h"
#include "dwdx.h"

namespace amici {
namespace model_icg_sd {

void dydx_icg_sd(realtype *dydx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *dwdx){
    dydx[0] = 1;
    dydx[26] = Fblood*Vre*(1 - HCT);
    dydx[27] = Fblood*Mr_icg*Vre*(1 - HCT);
    dydx[28] = Mr_icg;
    dydx[71] = 1;
    dydx[99] = Fblood*Vgi*(1 - HCT);
    dydx[100] = Fblood*Mr_icg*Vgi*(1 - HCT);
    dydx[101] = Mr_icg;
    dydx[142] = 1;
    dydx[172] = Fblood*Vli*(1 - HCT);
    dydx[173] = Fblood*Mr_icg*Vli*(1 - HCT);
    dydx[174] = Mr_icg;
    dydx[213] = 1;
    dydx[245] = Fblood*Vlu*(1 - HCT);
    dydx[246] = Fblood*Mr_icg*Vlu*(1 - HCT);
    dydx[247] = Mr_icg;
    dydx[284] = 1;
    dydx[318] = (1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);
    dydx[319] = Mr_icg*(1 - HCT)*(-BW*FVve*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVve);
    dydx[320] = Mr_icg;
    dydx[331] = -Ri_icg/(Mr_icg*std::pow(Cve_icg + 9.9999999999999998e-13, 2));
    dydx[355] = 1;
    dydx[391] = (1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);
    dydx[392] = Mr_icg*(1 - HCT)*(-BW*FVar*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVar);
    dydx[393] = Mr_icg;
    dydx[400] = (-Car_icg + Chv_icg)/std::pow(Car_icg + 9.9999999999999995e-8, 2) + 1.0/(Car_icg + 9.9999999999999995e-8);
    dydx[426] = 1;
    dydx[464] = (1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);
    dydx[465] = Mr_icg*(1 - HCT)*(-BW*FVpo*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVpo);
    dydx[466] = Mr_icg;
    dydx[497] = 1;
    dydx[537] = (1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);
    dydx[538] = Mr_icg*(1 - HCT)*(-BW*FVhv*Fblood*(-FVar - FVhv - FVpo - FVve + 1)/(FVar + FVhv + FVpo + FVve) + BW*FVhv);
    dydx[539] = Mr_icg;
    dydx[540] = -1/(Car_icg + 9.9999999999999995e-8);
    dydx[568] = 1;
    dydx[640] = 1;
    dydx[711] = 1;
    dydx[782] = 1;
    dydx[853] = 1;
}

} // namespace model_icg_sd
} // namespace amici
