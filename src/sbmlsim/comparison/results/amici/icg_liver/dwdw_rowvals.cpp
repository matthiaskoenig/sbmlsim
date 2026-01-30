#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 1> dwdw_rowvals_icg_liver_ = {
    1
};

void dwdw_rowvals_icg_liver(SUNMatrixWrapper &dwdw){
    dwdw.set_indexvals(gsl::make_span(dwdw_rowvals_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
