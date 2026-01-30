#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_sd {

static constexpr std::array<sunindextype, 1> dxdotdx_explicit_rowvals_icg_sd_ = {
    11
};

void dxdotdx_explicit_rowvals_icg_sd(SUNMatrixWrapper &dxdotdx_explicit){
    dxdotdx_explicit.set_indexvals(gsl::make_span(dxdotdx_explicit_rowvals_icg_sd_));
}
} // namespace model_icg_sd
} // namespace amici
