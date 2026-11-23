import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_jit.stubgen import _extract_cpp_ast_metadata, _postprocess_pyi  # noqa: E402


BINDING_SOURCE = '''
struct Widget {
    int size(int x) const;
};

void resize(Widget &self, int width, int height);

void bind(pybind11::module_ &module) {
    // The binding is held in a local variable (``auto held = ...``).
    auto held = py::class_<Widget>(module, "WidgetA");
    held.def("method", [](Widget &self, int x) { return self.size(x); });
    held.def("resize", &resize);
    // Declared with the class_ type directly (``py::class_<Widget> typed(...)``).
    py::class_<Widget> typed(module, "WidgetB");
    typed.def(py::init([](int x, int y) { return Widget(); }),
              py::arg("x"), py::arg("y"), "Make a widget.");
    // Chained bindings keep resolving.
    py::class_<Widget>(module, "WidgetC").def("method", [](Widget &self, int x) { return self.size(x); });
}
'''

STUB_SOURCE = '''\
class WidgetA:
    def method(self, arg0: int) -> int:
        ...
    def resize(self, arg0: int, arg1: int) -> None:
        ...
class WidgetB:
    def __init__(self, arg0: int, arg1: int) -> None:
        ...
class WidgetC:
    def method(self, arg0: int) -> int:
        ...
'''


def test_stubgen():
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_cpp  # noqa: F401
    except ImportError:
        print('tree-sitter-cpp not installed; skipping the stubgen AST test')
        return

    with tempfile.TemporaryDirectory(prefix='deep-jit-stubgen-') as directory:
        source = Path(directory) / 'bindings.cpp'
        source.write_text(BINDING_SOURCE, encoding='utf-8')

        cpp_funcs, py_to_cpp, py_cls_to_cpp = _extract_cpp_ast_metadata(Path(directory))

        # Variable-held class_ bindings resolve like chained ones, in both forms.
        assert py_cls_to_cpp['WidgetA'] == 'Widget'
        assert py_cls_to_cpp['WidgetB'] == 'Widget'
        assert py_cls_to_cpp['WidgetC'] == 'Widget'
        # The lambda receiver is dropped; only the real parameters are recorded,
        # under the class instead of the module scope.
        assert cpp_funcs[('WidgetA', 'method', 1)] == (['x'], None)
        assert cpp_funcs[('WidgetC', 'method', 1)] == (['x'], None)
        assert ('WidgetA', 'method', 2) not in cpp_funcs
        # py::init lambdas are recorded under __init__ with names and docstring.
        assert cpp_funcs[('WidgetB', '__init__', 2)] == (['x', 'y'], 'Make a widget.')
        # A free function bound as a method maps to the class for the argc+1 retry.
        assert py_to_cpp[('WidgetA', 'resize')] == ('Widget', 'resize')
        assert cpp_funcs[(None, 'resize', 3)] == (['self', 'width', 'height'], None)

        stub = Path(directory) / 'bindings.pyi'
        stub.write_text(STUB_SOURCE, encoding='utf-8')
        _postprocess_pyi(stub, cpp_funcs, py_to_cpp, py_cls_to_cpp)
        text = stub.read_text(encoding='utf-8')

        assert 'def method(self, x: int, /) -> int:' in text
        # The free function's receiver parameter is dropped in the visible signature.
        assert 'def resize(self, width: int, height: int, /) -> None:' in text
        assert 'def __init__(self, x: int, y: int, /) -> None:' in text
        assert '"""Make a widget."""' in text
        assert 'arg0' not in text


if __name__ == '__main__':
    test_stubgen()
