#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_sd {

static constexpr std::array<sunindextype, 73> dwdw_colptrs_icg_sd_ = {
    0, 1, 1, 1, 2, 2, 5, 6, 6, 6, 8, 8, 13, 15, 15, 15, 17, 19, 21, 23, 27, 27, 27, 27, 27, 27, 27, 29, 31, 33, 35, 35, 35, 35, 35, 38, 41, 44, 44, 44, 44, 44, 44, 44, 46, 47, 47, 47, 47, 51, 51, 51, 51, 51, 54, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56
};

void dwdw_colptrs_icg_sd(SUNMatrixWrapper &dwdw){
    dwdw.set_indexptrs(gsl::make_span(dwdw_colptrs_icg_sd_));
}
} // namespace model_icg_sd
} // namespace amici
