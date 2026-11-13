r"""Replace upstream malformed printf/abort handlers with Python-visible errors.

For pinned NKSR e403368 only. No reconstruction math or checkpoint changes.
The original CUDA source contains invalid \N escapes that crash CUDA 13 nvcc.
"""
from pathlib import Path

path = Path("package/csrc/sparse_solve/err_check.cu")
source = path.read_text()
start = source.index("inline void __cusparseSafeCall(")
end = source.index("void cusparseSafeCall(", start)
source = source[:start] + '''inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
{
    TORCH_CHECK(err == CUSPARSE_STATUS_SUCCESS,
                "CUSPARSE error: ", _cusparseGetErrorEnum(err), " at ", file, ":", line);
}

''' + source[end:]
start = source.index("void cublasSafeCall(")
source = source[:start] + '''void cublasSafeCall(cublasStatus_t err)
{
    TORCH_CHECK(err == CUBLAS_STATUS_SUCCESS, "CUBLAS error code: ", static_cast<int>(err));
}
'''
assert "cudaDeviceReset" not in source and "assert(0)" not in source
path.write_text(source)

# New pybind11 array_caster constructs from a nested initializer list, which
# NanoVDB's three-scalar constructors do not accept. Keep the existing cast
# direction and explicitly construct the three-coordinate input values.
path = Path("package/csrc/vdbops/PythonBindings.cpp")
source = path.read_text()
marker = "namespace pybind11 {namespace detail {"
helper = '''namespace pybind11 {namespace detail {
template <typename Vec, typename Scalar>
struct nksr_vec3_caster {
    PYBIND11_TYPE_CASTER(Vec, const_name("tuple"));
    bool load(handle src, bool convert) {
        type_caster<std::array<Scalar, 3>> caster;
        if (!caster.load(src, convert)) return false;
        auto a = cast_op<std::array<Scalar, 3>>(caster);
        this->value = Vec(a[0], a[1], a[2]);
        return true;
    }
    static handle cast(const Vec& src, return_value_policy, handle) {
        return pybind11::make_tuple(src[0], src[1], src[2]).release();
    }
};
}}

'''
assert source.count("array_caster<nanovdb::Coord, int32_t, false, 3>") == 1
source = source.replace(marker, helper + marker, 1)
source = source.replace("array_caster<nanovdb::Coord, int32_t, false, 3>",
                        "nksr_vec3_caster<nanovdb::Coord, int32_t>")
source = source.replace("array_caster<nanovdb::Vec3<ScalarT>, ScalarT, false, 3>",
                        "nksr_vec3_caster<nanovdb::Vec3<ScalarT>, ScalarT>")
path.write_text(source)
