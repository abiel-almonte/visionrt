#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Python.h>

#include "camera.hpp"
#include "graph.hpp"
#include "kernels.cuh"
#include "utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_visionrt, m) {
    m.doc() = "C++/CUDA runtime providing a GPU-resident pipeline for camera capture, preprocessing, and PyTorch integration.";

    py::class_<GraphExecutor>(m, "GraphExecutor", "Modfier that compiles and captures CUDA graph for the given PyTorch module")
        .def(py::init<py::object>(), py::arg("module"), "PyTorch module to compile and capture")
        .def("capture", &GraphExecutor::capture, py::arg("tensor"), "Capture CUDA graph")
        .def("__call__", &GraphExecutor::__call__, py::arg("tensor"), "Launch CUDA graph")
        .def("is_captured", &GraphExecutor::is_captured,"Return True if CUDA graph has been captured");


    py::class_<Camera>(m, "Camera", "Wrapper around a V4L2 camera device")
        .def(py::init<const char*>(), py::arg("device"), "Open a camera at the given device path (e.g., '/dev/video0').")
        .def("close", &Camera::close_camera, "Close the camera device.")
        .def("print_formats", &Camera::list_formats, "Print all supported camera formats.")
        .def("set_format", &Camera::set_format, py::arg("index"), "Set the capture format by index.")
        .def("print_selected_format", &Camera::print_format, "Print the currently selected format.")
        .def_property_readonly("width", &Camera::width, "Width of the current format.")
        .def_property_readonly("height", &Camera::height, "Height of the current format.")
        .def("__repr__", &Camera::__repr__)
        .def("__iter__", &Camera::__iter__, py::return_value_policy::reference_internal)
        .def("__next__", &Camera::__next__)
        .def("stream", &Camera::stream, py::return_value_policy::reference_internal, "Return an iterator that yields frames.");

    m.def("fused_add_relu_cuda", &launch_add_relu, "Fused add + relu");
    m.def("yuyv2rgb_cuda", &launch_yuyv2rgb_chw,
          py::arg("yuyv"), py::arg("height"), py::arg("width"),
          py::arg("scale"), py::arg("offset"),
          "YUYV to normalized RGB CHW conversion");
    m.def("set_verbose", &set_verbose, "Enable/disable verbose logging");

}
