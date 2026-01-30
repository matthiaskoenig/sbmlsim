#include <array>
#include <amici/defines.h>

namespace amici {

namespace model_icg_liver {

std::array<const char*, 7> parameterNames = {
    "Vmax for icg import", // p[0]
"Km icg of icg import", // p[1]
"Ki bilirubin of icg import", // p[2]
"scaling factor protein amount", // p[3]
"Vmax bile export icg", // p[4]
"Km bile export icg", // p[5]
"Vmax bile transport icg", // p[6]
};

std::array<const char*, 0> fixedParameterNames = {
    
};

std::array<const char*, 5> stateNames = {
    "icg (plasma)", // x_rdata[0]
"bilirubin (extern)", // x_rdata[1]
"icg (liver)", // x_rdata[2]
"icg (bile)", // x_rdata[3]
"icg (feces)", // x_rdata[4]
};

std::array<const char*, 9> observableNames = {
    "icg (plasma)", // y[0]
"bilirubin (extern)", // y[1]
"icg (liver)", // y[2]
"icg (bile)", // y[3]
"icg (feces)", // y[4]
"Vext", // y[5]
"Vli", // y[6]
"Vbi", // y[7]
"Vfeces", // y[8]
};

std::array<const ObservableScaling, 9> observableScalings = {
    ObservableScaling::lin, // y[0]
ObservableScaling::lin, // y[1]
ObservableScaling::lin, // y[2]
ObservableScaling::lin, // y[3]
ObservableScaling::lin, // y[4]
ObservableScaling::lin, // y[5]
ObservableScaling::lin, // y[6]
ObservableScaling::lin, // y[7]
ObservableScaling::lin, // y[8]
};

std::array<const char*, 4> expressionNames = {
    "bil_ext", // w[0]
"flux_ICGIM", // w[1]
"flux_ICGLI2CA", // w[2]
"flux_ICGLI2BI", // w[3]
};

std::array<const char*, 7> parameterIds = {
    "ICGIM_Vmax", // p[0]
"ICGIM_Km", // p[1]
"ICGIM_ki_bil", // p[2]
"f_oatp1b3", // p[3]
"ICGLI2CA_Vmax", // p[4]
"ICGLI2CA_Km", // p[5]
"ICGLI2BI_Vmax", // p[6]
};

std::array<const char*, 0> fixedParameterIds = {
    
};

std::array<const char*, 5> stateIds = {
    "icg_ext", // x_rdata[0]
"bil_ext", // x_rdata[1]
"icg", // x_rdata[2]
"icg_bi", // x_rdata[3]
"icg_feces", // x_rdata[4]
};

std::array<const char*, 9> observableIds = {
    "yicg_ext", // y[0]
"ybil_ext", // y[1]
"yicg", // y[2]
"yicg_bi", // y[3]
"yicg_feces", // y[4]
"yVext", // y[5]
"yVli", // y[6]
"yVbi", // y[7]
"yVfeces", // y[8]
};

std::array<const char*, 4> expressionIds = {
    "bil_ext", // w[0]
"flux_ICGIM", // w[1]
"flux_ICGLI2CA", // w[2]
"flux_ICGLI2BI", // w[3]
};

std::array<int, 4> stateIdxsSolver = {
    0, 2, 3, 4
};

std::array<bool, 0> rootInitialValues = {
    
};

} // namespace model_icg_liver

} // namespace amici
