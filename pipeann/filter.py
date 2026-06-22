from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from .C import AndSelector as _AndSelector
from .C import Attributes as _Attributes
from .C import LabelAndSelector as _LabelAndSelector
from .C import LabelOrSelector as _LabelOrSelector
from .C import NativeAttrIndex as _NativeAttrIndex
from .C import NativeAttrsVec as _NativeAttrsVec
from .C import NotSelector as _NotSelector
from .C import OrSelector as _OrSelector
from .C import RangeSelector as _RangeSelector
from .C import Selector as _Selector
from .C import StringEqSelector as _StringEqSelector
from .C import _save_attr_index_from_rows


class Attributes(_Attributes):
    """Python-friendly constructor for PipeANN attributes."""

    def __init__(self, data: Mapping[int, Sequence[int]] | _Attributes = {}):
        if isinstance(data, _Attributes):
            data = data.to_dict()
        super().__init__(data)

    def to_dict(self) -> Dict[int, List[int]]:
        return {
            int(key): [int(value) for value in values]
            for key, values in super().to_dict().items()
        }

    def __getitem__(self, key: int) -> List[int]:
        return list(super().get(int(key)))

    def __setitem__(self, key: int, values: Iterable[int]) -> None:
        super().set(int(key), [int(value) for value in values])

    def __contains__(self, key: int) -> bool:
        return bool(super().find(int(key)))

    def __repr__(self) -> str:
        return f"Attributes({self.to_dict()!r})"


class AttrsVec:
    """Row-oriented attribute container used for both build-time and query-time attrs."""

    def __init__(
        self,
        data: (
            _NativeAttrsVec
            | Sequence[Mapping[int, Sequence[int]] | Attributes]
        ) = [],
        attr_types: Mapping[int, str] = {},
    ) -> None:
        self.attr_types = {
            int(key): str(attr_type) for key, attr_type in attr_types.items()
        }
        
        if isinstance(data, _NativeAttrsVec):
            self._impl = data
            return

        self._impl = _NativeAttrsVec(
            row if isinstance(row, _Attributes) else Attributes(row) for row in data
        )

    def append(
        self,
        attrs: Mapping[int, Sequence[int]] | Attributes,
        attr_types: Mapping[int, str] = {},
    ) -> int:
        """Append one full row of attributes."""
        for key, attr_type in attr_types.items():
            self.attr_types[key] = attr_type

        row = attrs if isinstance(attrs, Attributes) else Attributes(attrs)
        self._impl.append(row)
        return len(self._impl) - 1

    def load_from_file(self, key: int, attr_type: str, filename: str | Path) -> None:
        """Load one query attribute file and merge it into the row-oriented container."""
        key = int(key)
        self.attr_types[key] = str(attr_type)
        self._impl.load_from_file(key, str(attr_type), str(filename))

    def save(self, key: int, filename: str | Path) -> None:
        """Save one attribute column as a native PipeANN attr index file."""
        key = int(key)
        rows = [attrs[key] for attrs in self]
        _save_attr_index_from_rows(rows, str(filename), self.attr_types[key])

    def attr_size(self) -> int:
        """Return the maximum serialized size of one attribute row."""
        max_size = 4
        for attrs in self:
            cur_size = 4
            for values in attrs.to_dict().values():
                cur_size += 8 + 4 * len(values)
            max_size = max(max_size, cur_size)
        return max_size

    def to_list(self) -> list[dict[int, list[int]]]:
        return [attrs.to_dict() for attrs in self]

    def __len__(self) -> int:
        return len(self._impl)

    def __getitem__(self, index: int) -> Attributes:
        return Attributes(self._impl[int(index)])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"AttrsVec(attr_types={self.attr_types!r}, rows={self.to_list()!r})"


# Make type checkers happy about the native pybind classes.
if TYPE_CHECKING:

    class NativeAttrIndex:
        attr_type: str
        n_vectors: int

    class Selector:
        def __init__(self) -> None: ...

        def estimate_selectivity(self, query_attrs: Attributes) -> float: ...

        def estimate_precision(self, query_attrs: Attributes) -> float: ...

        def estimate_prefilter_reads(self, query_attrs: Attributes) -> int: ...

        def pre_filter(self, query_attrs: Attributes) -> List[int]: ...

        def is_member(
            self, target_id: int, query_attrs: Attributes, target_attrs: Attributes
        ) -> bool: ...

        def estimate_infilter_reads(self, query_attrs: Attributes) -> int: ...

        def prepare_in_filter(self, query_attrs: Attributes) -> List[int]: ...

        def is_member_approx(
            self, target_id: int, query_attrs: Attributes, vector_id_list: Sequence[int]
        ) -> bool: ...

    class LabelOrSelector(Selector):
        def __init__(
            self, key: int, base_key: int, attr_index: NativeAttrIndex
        ) -> None: ...

    class LabelAndSelector(Selector):
        def __init__(
            self, key: int, base_key: int, attr_index: NativeAttrIndex
        ) -> None: ...

    class RangeSelector(Selector):
        def __init__(
            self, key: int, base_key: int, attr_index: NativeAttrIndex
        ) -> None: ...

    class AndSelector(Selector):
        def __init__(self, *children: Selector) -> None: ...

    class OrSelector(Selector):
        def __init__(self, *children: Selector) -> None: ...

    class NotSelector(Selector):
        def __init__(self, child: Selector, n_vectors: int) -> None: ...

else:
    NativeAttrIndex = _NativeAttrIndex
    Selector = _Selector
    LabelOrSelector = _LabelOrSelector
    LabelAndSelector = _LabelAndSelector
    RangeSelector = _RangeSelector
    StringEqSelector = _StringEqSelector
    AndSelector = _AndSelector
    OrSelector = _OrSelector
    NotSelector = _NotSelector


def pack_string(s: str) -> List[int]:
    """Pack a UTF-8 string into a packed-bytes Attribute (no length prefix,
    NUL-padded to a uint32 boundary). Mirrors the C++ side of `string` attr
    index storage. Embedded NUL bytes are not allowed.
    """
    if "\x00" in s:
        raise ValueError("string attribute cannot contain embedded NUL byte")
    b = s.encode("utf-8")
    pad = (-len(b)) % 4
    b += b"\x00" * pad
    return [int.from_bytes(b[i:i + 4], "little") for i in range(0, len(b), 4)]


def unpack_string(packed: Sequence[int]) -> str:
    """Inverse of ``pack_string``: decode a packed-bytes attribute back to str."""
    raw = b"".join(int(x).to_bytes(4, "little") for x in packed)
    end = raw.find(b"\x00")
    if end != -1:
        raw = raw[:end]
    return raw.decode("utf-8")


__all__ = [
    "AndSelector",
    "AttrsVec",
    "Attributes",
    "LabelAndSelector",
    "LabelOrSelector",
    "NativeAttrIndex",
    "NotSelector",
    "OrSelector",
    "RangeSelector",
    "Selector",
    "StringEqSelector",
    "pack_string",
    "unpack_string",
]
