#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_sd {

static constexpr std::array<sunindextype, 36> dxdotdp_explicit_colptrs_icg_sd_ = {
    0, 0, 0, 10, 10, 10, 19, 27, 29, 31, 34, 36, 41, 46, 51, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 59, 59, 59, 61, 61, 61, 61, 61, 61, 61, 61
};

void dxdotdp_explicit_colptrs_icg_sd(SUNMatrixWrapper &dxdotdp_explicit){
    dxdotdp_explicit.set_indexptrs(gsl::make_span(dxdotdp_explicit_colptrs_icg_sd_));
}
} // namespace model_icg_sd
} // namespace amici
