#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_sd {

static constexpr std::array<sunindextype, 27> dwdx_rowvals_icg_sd_ = {
    43, 58, 27, 60, 28, 65, 69, 29, 68, 2, 18, 67, 4, 15, 57, 59, 61, 62, 17, 63, 64, 4, 16, 66, 70, 71, 56
};

void dwdx_rowvals_icg_sd(SUNMatrixWrapper &dwdx){
    dwdx.set_indexvals(gsl::make_span(dwdx_rowvals_icg_sd_));
}
} // namespace model_icg_sd
} // namespace amici
