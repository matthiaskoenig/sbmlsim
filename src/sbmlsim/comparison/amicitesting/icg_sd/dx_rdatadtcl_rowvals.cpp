#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_sd {

static constexpr std::array<sunindextype, 1> dx_rdatadtcl_rowvals_icg_sd_ = {
    9
};

void dx_rdatadtcl_rowvals_icg_sd(SUNMatrixWrapper &dx_rdatadtcl){
    dx_rdatadtcl.set_indexvals(gsl::make_span(dx_rdatadtcl_rowvals_icg_sd_));
}
} // namespace model_icg_sd
} // namespace amici
