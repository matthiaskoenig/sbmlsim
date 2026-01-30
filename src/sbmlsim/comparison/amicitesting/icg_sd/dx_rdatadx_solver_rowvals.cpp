#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_sd {

static constexpr std::array<sunindextype, 13> dx_rdatadx_solver_rowvals_icg_sd_ = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13
};

void dx_rdatadx_solver_rowvals_icg_sd(SUNMatrixWrapper &dx_rdatadx_solver){
    dx_rdatadx_solver.set_indexvals(gsl::make_span(dx_rdatadx_solver_rowvals_icg_sd_));
}
} // namespace model_icg_sd
} // namespace amici
