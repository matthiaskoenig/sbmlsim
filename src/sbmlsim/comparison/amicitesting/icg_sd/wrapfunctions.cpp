#include "amici/model.h"
#include "wrapfunctions.h"
#include "icg_sd.h"

namespace amici {
namespace generic_model {

std::unique_ptr<amici::Model> getModel() {
    return std::unique_ptr<amici::Model>(
        new amici::model_icg_sd::Model_icg_sd());
}


} // namespace generic_model

} // namespace amici
