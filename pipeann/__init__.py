# The native extension (.C) backs the in-process API. Importing it can fail on
# a machine where the shared library is missing or mislinked. That must not stop
# `pipeann-server`, which only launches the standalone server binary (see
# pipeann/server.py), so defer the failure to first use of the in-process API
# rather than at package import.
try:
    from .index import IndexPipeANN, Metric, VALID_DATA_TYPES
    from .filter import (
        AndSelector,
        AttrsVec,
        Attributes,
        LabelAndSelector,
        LabelOrSelector,
        NativeAttrIndex,
        NotSelector,
        OrSelector,
        RangeSelector,
        Selector,
        StringEqSelector,
        pack_string,
        unpack_string,
    )
    from .milvus import MilvusClient, DataType
except ImportError as _exc:  # native extension unavailable
    _NATIVE_IMPORT_ERROR = _exc

    # Names that live in the native-backed in-process API. Only these should
    # report the underlying load failure; everything else (e.g. the `server`
    # submodule, which needs no native code) falls through to normal lookup.
    _NATIVE_API_NAMES = frozenset({
        "IndexPipeANN", "Metric", "VALID_DATA_TYPES",
        "AndSelector", "AttrsVec", "Attributes", "LabelAndSelector",
        "LabelOrSelector", "NativeAttrIndex", "NotSelector", "OrSelector",
        "RangeSelector", "Selector", "StringEqSelector",
        "pack_string", "unpack_string",
        "MilvusClient", "DataType",
    })

    def __getattr__(name):
        if name in _NATIVE_API_NAMES:
            raise ImportError(
                f"pipeann.{name} requires the native extension, which failed "
                f"to load: {_NATIVE_IMPORT_ERROR}"
            ) from _NATIVE_IMPORT_ERROR
        raise AttributeError(f"module 'pipeann' has no attribute '{name}'")
