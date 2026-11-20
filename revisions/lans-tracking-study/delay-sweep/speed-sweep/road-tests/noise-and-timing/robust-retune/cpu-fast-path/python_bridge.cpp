#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

extern "C" {
struct BoundActor;
double steer_lqr(const double* state, const double* gains);
double steer_bound_actor(const BoundActor* context, const double* state, int tick,
                         double previous_delay, double mean_delay);
#ifdef LANS_FIXED_DELAY_PLAN
struct FixedActor;
double steer_fixed_actor(const FixedActor* actor, const double* state, int tick);
#endif
}

namespace {
class StateView {
    Py_buffer buffer{};
    bool acquired = false;

public:
    explicit StateView(PyObject* state) {
        acquired = PyObject_GetBuffer(state, &buffer, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0;
    }

    ~StateView() {
        if (acquired)
            PyBuffer_Release(&buffer);
    }

    const double* data() const {
        if (!acquired)
            return nullptr;
        if (buffer.ndim != 1 || buffer.len != 4 * sizeof(double) ||
            buffer.itemsize != sizeof(double) || std::strcmp(buffer.format, "d") != 0 ||
            reinterpret_cast<std::uintptr_t>(buffer.buf) % alignof(double) != 0) {
            PyErr_SetString(PyExc_TypeError, "state must be four contiguous native float64 values");
            return nullptr;
        }
        return static_cast<const double*>(buffer.buf);
    }
};

PyObject* lqr(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "lqr expects the bound gain address and controller state");
        return nullptr;
    }
    const auto* gains = static_cast<const double*>(PyCapsule_GetPointer(args[0], "lans.gains"));
    if (gains == nullptr)
        return nullptr;
    const StateView state(args[1]);
    const double* x = state.data();
    return x == nullptr ? nullptr : PyFloat_FromDouble(steer_lqr(x, gains));
}

PyObject* actor(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "actor expects the bound context, state, tick and delay observations");
        return nullptr;
    }
    const auto* context = static_cast<const BoundActor*>(PyCapsule_GetPointer(args[0], "lans.actor"));
    if (context == nullptr)
        return nullptr;
    const long tick = PyLong_AsLong(args[2]);
    const double delay = PyFloat_AsDouble(args[3]);
    const double mean = PyFloat_AsDouble(args[4]);
    if (PyErr_Occurred())
        return nullptr;
    if (tick < 0 || tick > std::numeric_limits<int>::max()) {
        PyErr_SetString(PyExc_ValueError, "tick is outside the bound road");
        return nullptr;
    }
    const StateView state(args[1]);
    const double* x = state.data();
    if (x == nullptr)
        return nullptr;
    const double command = steer_bound_actor(context, x, static_cast<int>(tick), delay, mean);
    if (!std::isfinite(command)) {
        PyErr_SetString(PyExc_ValueError, "controller inputs produced a non-finite command or exceeded the bound road");
        return nullptr;
    }
    return PyFloat_FromDouble(command);
}

#ifdef LANS_FIXED_DELAY_PLAN
PyObject* fixed_actor(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "fixed_actor expects the fixed context, state and tick");
        return nullptr;
    }
    const auto* context = static_cast<const FixedActor*>(PyCapsule_GetPointer(args[0], "lans.fixed"));
    if (context == nullptr)
        return nullptr;
    const long tick = PyLong_AsLong(args[2]);
    if (PyErr_Occurred())
        return nullptr;
    if (tick < 0 || tick > std::numeric_limits<int>::max()) {
        PyErr_SetString(PyExc_ValueError, "tick is outside the bound road");
        return nullptr;
    }
    const StateView state(args[1]);
    const double* x = state.data();
    if (x == nullptr)
        return nullptr;
    const double command = steer_fixed_actor(context, x, static_cast<int>(tick));
    if (!std::isfinite(command)) {
        PyErr_SetString(PyExc_ValueError, "controller inputs produced a non-finite command or exceeded the bound road");
        return nullptr;
    }
    return PyFloat_FromDouble(command);
}
#endif

PyMethodDef methods[] = {
    {"lqr", reinterpret_cast<PyCFunction>(lqr), METH_FASTCALL, "Apply the bound LQR gain."},
    {"actor", reinterpret_cast<PyCFunction>(actor), METH_FASTCALL, "Apply a bound neural controller."},
#ifdef LANS_FIXED_DELAY_PLAN
    {"fixed_actor", reinterpret_cast<PyCFunction>(fixed_actor), METH_FASTCALL, "Apply a fixed-delay neural controller."},
#endif
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "lans_bound", "Bound steering controllers", -1, methods, nullptr, nullptr, nullptr, nullptr
};
}

PyMODINIT_FUNC PyInit_lans_bound() {
    return PyModule_Create(&module);
}
