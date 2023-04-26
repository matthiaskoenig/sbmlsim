#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 5> dx_rdatadx_solver_colptrs_icg_liver_ = {
    0, 1, 2, 3, 4
};

void dx_rdatadx_solver_colptrs_icg_liver(SUNMatrixWrapper &dx_rdatadx_solver){
    dx_rdatadx_solver.set_indexptrs(gsl::make_span(dx_rdatadx_solver_colptrs_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
