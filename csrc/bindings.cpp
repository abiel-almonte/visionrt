#include <pybind11/pybind11.h>
#include <Python.h>

#include "camera.hpp"
#include "graph.hpp"
#include "ops.hpp"
#include "preprocessing.hpp"
#include "utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_visionrt, m) {
    m.doc() = "C++/CUDA runtime providing a GPU-resident pipeline for camera capture, preprocessing, and PyTorch integration.";

    py::class_<GraphExecutor>(m, "GraphExecutor", "Modfier that compiles and captures CUDA graph for the given PyTorch module")
        .def(py::init<py::object>(), py::arg("module"), "PyTorch module to compile and capture")
        .def("capture", &GraphExecutor::capture, py::arg("tensor"), "Capture CUDA graph")
        .def("__call__", &GraphExecutor::__call__, py::arg("tensor"), "Launch CUDA graph")
        .def("is_captured", &GraphExecutor::is_captured,"Return True if CUDA graph has been captured");


    py::class_<FrameGenerator>(m, "FrameGenerator", "Iterator that yields preprocessed frames from a Camera as torch.Tensors")
        .def("__iter__", &FrameGenerator::__iter__, py::return_value_policy::reference_internal, "Return the iterator object itself.")
        .def("__next__", &FrameGenerator::__next__, "Advance to the next frame.\n\n" "Raises StopIteration when no frames remain.");

    py::class_<Camera>(m, "Camera", "Wrapper around a V4L2 camera device")
        .def(py::init<const char*>(), py::arg("device"), "Open a camera at the given device path (e.g., '/dev/video0').")

        .def("close", &Camera::close_camera, "Close the opened camera")
        .def("reset_stats", &Camera::reset_stats, "Reset internal timing statistics.")
        .def("print_formats", &Camera::list_formats, "Print all supported camera formats.")
        .def("set_format", &Camera::set_format, py::arg("index"), "Set the capture format.")
        .def("print_selected_format", &Camera::print_format, "Print the currently selected camera format.")
        .def("__repr__", &Camera::__repr__, "Print the Camera object.")
        .def("stream", [](Camera& cam) {
            return FrameGenerator(&cam);
        }, py::return_value_policy::move,
           "Return a FrameGenerator that yields frames from this Camera.");

    m.def("fused_add_relu_cuda", &launch_add_relu, "Fused add + relu");
    m.def("set_verbose", &set_verbose, "Enable/disable verbose logging");

}
