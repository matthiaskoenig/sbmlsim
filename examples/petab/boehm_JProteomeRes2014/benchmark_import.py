import h5py


class DataProvider:
    def __init__(self, h5_file):
        self.h5_file = h5_file

    def get_edata(self):
        pass

    def get_timepoints(self):
        with h5py.File(self.h5_file, "r") as f:
            return f["/amiciOptions/ts"][:]

    def get_pscales(self):
        with h5py.File(self.h5_file, "r") as f:
            return f["/amiciOptions/pscale"][:]

    def get_fixed_parameters(self):
        with h5py.File(self.h5_file, "r") as f:
            fixed_parameters = f["/fixedParameters/k"][:]
            return fixed_parameters[0]

    def get_fixed_parameters_names(self):
        with h5py.File(self.h5_file, "r") as f:
            return f["/fixedParameters/parameterNames"][:]

    def get_initial_states(self):
        pass

    def get_measurements(self):
        with h5py.File(self.h5_file, "r") as f:
            return f["/measurements/y"][:]

    def get_ysigma(self):
        with h5py.File(self.h5_file, "r") as f:
            return f["/measurements/ysigma"][:]

    def get_observableNames(self):
        with h5py.File(self.h5_file, "r") as f:
            return f["/measurements/observableNames"]
