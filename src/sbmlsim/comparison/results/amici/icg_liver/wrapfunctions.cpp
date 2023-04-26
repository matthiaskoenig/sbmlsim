#include "amici/model.h"
#include "wrapfunctions.h"
#include "icg_liver.h"

namespace amici {
namespace generic_model {

std::unique_ptr<amici::Model> getModel() {
    return std::unique_ptr<amici::Model>(
        new amici::model_icg_liver::Model_icg_liver());
}


} // namespace generic_model

} // namespace amici
