#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_sd {

static constexpr std::array<sunindextype, 56> dwdw_rowvals_icg_sd_ = {
    69, 19, 26, 37, 38, 56, 27, 45, 28, 46, 69, 70, 71, 29, 47, 30, 39, 31, 40, 32, 41, 33, 42, 34, 35, 36, 54, 43, 52, 45, 49, 46, 50, 47, 51, 48, 59, 60, 44, 53, 66, 44, 67, 68, 52, 55, 54, 53, 63, 64, 65, 61, 62, 65, 57, 58
};

void dwdw_rowvals_icg_sd(SUNMatrixWrapper &dwdw){
    dwdw.set_indexvals(gsl::make_span(dwdw_rowvals_icg_sd_));
}
} // namespace model_icg_sd
} // namespace amici
