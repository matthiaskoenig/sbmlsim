#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "tcl.h"
#include "p.h"
#include "dx_rdatadtcl.h"

namespace amici {
namespace model_icg_sd {

void dx_rdatadtcl_icg_sd(realtype *dx_rdatadtcl, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k){
    dLI__bil_ext_dtcl_LI__bil_ext = 1;  // dx_rdatadtcl[0]
}

} // namespace model_icg_sd
} // namespace amici
