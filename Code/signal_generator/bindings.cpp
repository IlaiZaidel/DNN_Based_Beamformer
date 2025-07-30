// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SignalGenerator.h"

namespace py = pybind11;

PYBIND11_MODULE(signal_generator, m) {
    py::class_<SignalGenerator::Result>(m, "Result")
        .def_readonly("output", &SignalGenerator::Result::output)
        .def_readonly("beta_hat", &SignalGenerator::Result::beta_hat);

    py::class_<SignalGenerator>(m, "SignalGenerator")
        .def(py::init<>())
        .def("generate", &SignalGenerator::generate,
             py::arg("input_signal"),
             py::arg("c"),
             py::arg("fs"),
             py::arg("r_path"),
             py::arg("s_path"),
             py::arg("L"),
             py::arg("beta_or_tr"),
             py::arg("nsamples") = -1,
             py::arg("mtype") = "o",
             py::arg("order") = -1,
             py::arg("dim") = 3,
             py::arg("orientation") = std::vector<std::vector<double>>(),
             py::arg("hp_filter") = true);
}
