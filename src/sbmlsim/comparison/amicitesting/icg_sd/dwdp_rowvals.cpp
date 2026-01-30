#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_sd {

static constexpr std::array<sunindextype, 196> dwdp_rowvals_icg_sd_ = {
    1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 37, 38, 1, 3, 7, 10, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 37, 38, 43, 45, 46, 47, 52, 69, 70, 71, 7, 10, 13, 14, 15, 16, 17, 18, 20, 22, 24, 27, 28, 29, 30, 31, 32, 33, 37, 43, 45, 46, 47, 52, 5, 9, 20, 21, 5, 8, 5, 11, 22, 23, 5, 12, 24, 25, 5, 7, 10, 13, 14, 15, 16, 17, 18, 30, 31, 32, 33, 5, 7, 10, 13, 14, 15, 16, 17, 18, 30, 31, 32, 33, 5, 7, 10, 13, 14, 15, 16, 17, 18, 30, 31, 32, 33, 5, 7, 10, 13, 14, 15, 16, 17, 18, 30, 31, 32, 33, 34, 35, 36, 61, 62, 63, 64, 65, 23, 69, 70, 71, 34, 35, 3, 35, 11, 22, 23, 2, 30, 31, 32, 33, 39, 40, 41, 42, 45, 46, 47, 49, 50, 51, 52, 55, 56, 6, 2, 69, 69, 69, 69, 70, 70, 71
};

void dwdp_rowvals_icg_sd(SUNMatrixWrapper &dwdp){
    dwdp.set_indexvals(gsl::make_span(dwdp_rowvals_icg_sd_));
}
} // namespace model_icg_sd
} // namespace amici
