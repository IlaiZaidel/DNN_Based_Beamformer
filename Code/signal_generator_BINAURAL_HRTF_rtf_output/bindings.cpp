// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SignalGenerator.h"

namespace py = pybind11;

PYBIND11_MODULE(signal_generator, m) {
    py::class_<SignalGenerator::Result>(m, "Result")
        .def_readonly("output", &SignalGenerator::Result::output)
        .def_readonly("beta_hat", &SignalGenerator::Result::beta_hat)
        .def_readonly("all_rirs", &SignalGenerator::Result::all_rirs);

    py::class_<SignalGenerator>(m, "SignalGenerator")
        .def(py::init<>())
        .def("generate_hrtf", &SignalGenerator::generate_hrtf,
             py::arg("input_signal"),
             py::arg("s_path"),
             py::arg("r_head_path"),
             py::arg("hrirs"),
             py::arg("azelevs"),
             py::arg("stft_win") = 512,
             py::arg("stft_hop") = 128,
             py::arg("update_every") = 128);
}
