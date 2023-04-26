#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_icg_liver {

static constexpr std::array<sunindextype, 5> dwdw_colptrs_icg_liver_ = {
    0, 1, 1, 1, 1
};

void dwdw_colptrs_icg_liver(SUNMatrixWrapper &dwdw){
    dwdw.set_indexptrs(gsl::make_span(dwdw_colptrs_icg_liver_));
}
} // namespace model_icg_liver
} // namespace amici
