#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "tcl.h"
#include "p.h"

namespace amici {
namespace model_icg_liver {

void x_rdata_icg_liver(realtype *x_rdata, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k){
    x_rdata[0] = icg_ext;
    x_rdata[1] = tcl_bil_ext;
    x_rdata[2] = icg;
    x_rdata[3] = icg_bi;
    x_rdata[4] = icg_feces;
}

} // namespace model_icg_liver
} // namespace amici
