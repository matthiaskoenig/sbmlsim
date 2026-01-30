#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "x.h"
#include "p.h"
#include "w.h"

namespace amici {
namespace model_icg_liver {

void y_icg_liver(realtype *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w){
    y[0] = icg_ext;
    y[1] = bil_ext;
    y[2] = icg;
    y[3] = icg_bi;
    y[4] = icg_feces;
    y[5] = 4.0;
    y[6] = 1.5;
    y[7] = 1.0;
    y[8] = 1.0;
}

} // namespace model_icg_liver
} // namespace amici
