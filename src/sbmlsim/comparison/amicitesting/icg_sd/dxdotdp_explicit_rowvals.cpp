#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_sd {

static constexpr std::array<sunindextype, 61> dxdotdp_explicit_rowvals_icg_sd_ = {
    0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 9, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 0, 10, 0, 2, 9, 0, 3, 0, 4, 5, 6, 7, 0, 4, 5, 6, 7, 0, 4, 5, 6, 7, 0, 4, 5, 6, 7, 9, 2, 9, 11, 12
};

void dxdotdp_explicit_rowvals_icg_sd(SUNMatrixWrapper &dxdotdp_explicit){
    dxdotdp_explicit.set_indexvals(gsl::make_span(dxdotdp_explicit_rowvals_icg_sd_));
}
} // namespace model_icg_sd
} // namespace amici
