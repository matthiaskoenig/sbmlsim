#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_sd {

static constexpr std::array<sunindextype, 14> dxdotdx_explicit_colptrs_icg_sd_ = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1
};

void dxdotdx_explicit_colptrs_icg_sd(SUNMatrixWrapper &dxdotdx_explicit){
    dxdotdx_explicit.set_indexptrs(gsl::make_span(dxdotdx_explicit_colptrs_icg_sd_));
}
} // namespace model_icg_sd
} // namespace amici
