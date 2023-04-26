#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_sd {

static constexpr std::array<sunindextype, 2> dx_rdatadtcl_colptrs_icg_sd_ = {
    0, 1
};

void dx_rdatadtcl_colptrs_icg_sd(SUNMatrixWrapper &dx_rdatadtcl){
    dx_rdatadtcl.set_indexptrs(gsl::make_span(dx_rdatadtcl_colptrs_icg_sd_));
}
} // namespace model_icg_sd
} // namespace amici
