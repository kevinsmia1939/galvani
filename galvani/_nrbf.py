"""Small, safe decoder for the records used by NOVA BinaryFormatter files.

This module decodes the wire representation into dictionaries and arrays.  It
does not load .NET assemblies or instantiate any of the serialized classes.
"""

# SPDX-FileCopyrightText: 2026 OpenAI
# SPDX-License-Identifier: GPL-3.0-or-later

import struct

import numpy as np


class NRBFError(ValueError):
    """Raised when an invalid or unsupported NRBF stream is encountered."""


_RECORD_HEADER = 0
_RECORD_CLASS_WITH_ID = 1
_RECORD_SYSTEM_CLASS = 4
_RECORD_CLASS = 5
_RECORD_STRING = 6
_RECORD_BINARY_ARRAY = 7
_RECORD_PRIMITIVE = 8
_RECORD_REFERENCE = 9
_RECORD_NULL = 10
_RECORD_END = 11
_RECORD_LIBRARY = 12
_RECORD_NULL_256 = 13
_RECORD_NULL_MULTIPLE = 14
_RECORD_PRIMITIVE_ARRAY = 15
_RECORD_OBJECT_ARRAY = 16
_RECORD_STRING_ARRAY = 17

_TYPE_PRIMITIVE = 0
_TYPE_SYSTEM_CLASS = 3
_TYPE_CLASS = 4
_TYPE_PRIMITIVE_ARRAY = 7

_ARRAY_OFFSET_TYPES = (3, 4, 5)

# These value-type arrays dominate the size of high-rate NOVA files.  NOVA
# also serializes the corresponding calculated signal arrays, which are what
# its own public API exposes.  Decode the records but do not retain them.
_DISCARDED_ARRAY_CLASSES = (
    "EcoChemie.Autolab.Adc164RawSample",
    "EcoChemie.Autolab.Adc164RawCrSample",
)

_NUMPY_PRIMITIVES = {
    1: "?",       # Boolean
    2: "u1",      # Byte
    6: "<f8",     # Double
    7: "<i2",     # Int16
    8: "<i4",     # Int32
    9: "<i8",     # Int64
    10: "i1",     # SByte
    11: "<f4",    # Single
    12: "<i8",    # TimeSpan
    13: "<u8",    # DateTime
    14: "<u2",    # UInt16
    15: "<u4",    # UInt32
    16: "<u8",    # UInt64
}


class NRBFParser:
    """Decode one NRBF stream starting at *offset* in a bytes-like object."""

    def __init__(self, data, offset=0):
        self.data = memoryview(data)
        self.offset = offset
        self.objects = {}
        self.libraries = {}
        self.class_definitions = {}

    def parse(self):
        """Return the unresolved root object and stop just after MessageEnd."""
        if self._u8() != _RECORD_HEADER:
            raise self._error("stream does not start with an NRBF header")
        root_id = self._i32()
        self._i32()  # header id
        major = self._i32()
        minor = self._i32()
        if (major, minor) != (1, 0):
            raise self._error("unsupported NRBF version %d.%d" % (major, minor))

        while self.offset < len(self.data):
            record_offset = self.offset
            record = self._u8()
            if record == _RECORD_END:
                return self.objects.get(root_id)
            if record == _RECORD_LIBRARY:
                self._read_library()
            elif record == _RECORD_CLASS:
                self._read_class(False)
            elif record == _RECORD_SYSTEM_CLASS:
                self._read_class(True)
            elif record == _RECORD_CLASS_WITH_ID:
                self._read_class_with_id()
            elif record == _RECORD_STRING:
                self._read_string()
            elif record == _RECORD_PRIMITIVE_ARRAY:
                self._read_primitive_array()
            elif record == _RECORD_OBJECT_ARRAY:
                self._read_object_array()
            elif record == _RECORD_STRING_ARRAY:
                self._read_object_array()
            elif record == _RECORD_BINARY_ARRAY:
                self._read_binary_array()
            else:
                raise self._error(
                    "unsupported top-level record %d at offset %d"
                    % (record, record_offset)
                )
        raise self._error("NRBF stream has no MessageEnd record")

    def resolve(self, value):
        """Resolve one MemberReference, without recursively copying a graph."""
        seen = set()
        while isinstance(value, dict) and set(value) == {"__ref__"}:
            object_id = value["__ref__"]
            if object_id in seen:
                return None
            seen.add(object_id)
            value = self.objects.get(object_id)
        return value

    def _error(self, message):
        return NRBFError("%s (offset %d)" % (message, self.offset))

    def _read(self, size):
        end = self.offset + size
        if size < 0 or end > len(self.data):
            raise self._error("unexpected end of file")
        result = self.data[self.offset:end]
        self.offset = end
        return result

    def _unpack(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack_from(fmt, self._read(size))[0]

    def _u8(self):
        return self._unpack("<B")

    def _i8(self):
        return self._unpack("<b")

    def _u16(self):
        return self._unpack("<H")

    def _i16(self):
        return self._unpack("<h")

    def _u32(self):
        return self._unpack("<I")

    def _i32(self):
        return self._unpack("<i")

    def _u64(self):
        return self._unpack("<Q")

    def _i64(self):
        return self._unpack("<q")

    def _f32(self):
        return self._unpack("<f")

    def _f64(self):
        return self._unpack("<d")

    def _string(self):
        length = 0
        shift = 0
        while True:
            byte = self._u8()
            length |= (byte & 0x7f) << shift
            if not byte & 0x80:
                break
            shift += 7
            if shift > 35:
                raise self._error("invalid length-prefixed string")
        try:
            return self._read(length).tobytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise self._error("invalid UTF-8 string: %s" % exc)

    def _primitive(self, primitive_type):
        readers = {
            1: self._u8,
            2: self._u8,
            6: self._f64,
            7: self._i16,
            8: self._i32,
            9: self._i64,
            10: self._i8,
            11: self._f32,
            12: self._i64,
            13: self._u64,
            14: self._u16,
            15: self._u32,
            16: self._u64,
        }
        if primitive_type == 1:
            return bool(self._u8())
        if primitive_type == 3:  # Char is UTF-8 encoded by BinaryFormatter.
            first = self._u8()
            if first < 0x80:
                return chr(first)
            count = 1 if first < 0xe0 else 2 if first < 0xf0 else 3
            raw = bytes([first] + [self._u8() for unused in range(count)])
            return raw.decode("utf-8")
        if primitive_type == 5:  # Decimal is written as a string.
            return self._string()
        try:
            return readers[primitive_type]()
        except KeyError:
            raise self._error("unsupported primitive type %d" % primitive_type)

    def _read_library(self):
        library_id = self._i32()
        self.libraries[library_id] = self._string()

    def _read_class_info(self):
        object_id = self._i32()
        name = self._string()
        count = self._i32()
        if count < 0 or count > 100000:
            raise self._error("invalid class member count %d" % count)
        names = [self._string() for unused in range(count)]
        return object_id, name, names

    def _read_type_info(self, count):
        binary_types = [self._u8() for unused in range(count)]
        additional = []
        for binary_type in binary_types:
            if binary_type in (_TYPE_PRIMITIVE, _TYPE_PRIMITIVE_ARRAY):
                additional.append(self._u8())
            elif binary_type == _TYPE_SYSTEM_CLASS:
                additional.append(self._string())
            elif binary_type == _TYPE_CLASS:
                additional.append((self._string(), self._i32()))
            elif binary_type in (1, 2, 5, 6):
                additional.append(None)
            else:
                raise self._error("unsupported binary type %d" % binary_type)
        return binary_types, additional

    def _read_class(self, system_class):
        object_id, name, names = self._read_class_info()
        binary_types, additional = self._read_type_info(len(names))
        if not system_class:
            self._i32()  # library id
        definition = (name, names, binary_types, additional)
        self.class_definitions[object_id] = definition
        obj = self._read_class_value(object_id, definition)
        return obj

    def _read_class_with_id(self):
        object_id = self._i32()
        metadata_id = self._i32()
        try:
            definition = self.class_definitions[metadata_id]
        except KeyError:
            raise self._error("unknown class metadata id %d" % metadata_id)
        return self._read_class_value(object_id, definition)

    def _read_class_value(self, object_id, definition):
        name, names, binary_types, additional = definition
        discard = name in _DISCARDED_ARRAY_CLASSES
        if discard:
            for binary_type, extra in zip(binary_types, additional):
                self._read_value(binary_type, extra)
            return None
        values = [
            self._read_value(binary_type, extra)
            for binary_type, extra in zip(binary_types, additional)
        ]
        obj = dict(zip(names, values))
        obj["__class__"] = name
        self.objects[object_id] = obj
        return obj

    def _read_value(self, binary_type, additional):
        if binary_type == _TYPE_PRIMITIVE:
            return self._primitive(additional)
        return self._read_inline()

    def _read_inline(self):
        record_offset = self.offset
        record = self._u8()
        if record == _RECORD_STRING:
            return self._read_string()
        if record == _RECORD_REFERENCE:
            return {"__ref__": self._i32()}
        if record == _RECORD_NULL:
            return None
        if record == _RECORD_CLASS:
            obj = self._read_class(False)
            return obj
        if record == _RECORD_SYSTEM_CLASS:
            return self._read_class(True)
        if record == _RECORD_CLASS_WITH_ID:
            return self._read_class_with_id()
        if record == _RECORD_PRIMITIVE_ARRAY:
            return self._read_primitive_array()
        if record in (_RECORD_OBJECT_ARRAY, _RECORD_STRING_ARRAY):
            return self._read_object_array()
        if record == _RECORD_BINARY_ARRAY:
            return self._read_binary_array()
        if record == _RECORD_PRIMITIVE:
            return self._primitive(self._u8())
        if record == _RECORD_LIBRARY:
            self._read_library()
            return self._read_inline()
        raise self._error(
            "unsupported inline record %d at offset %d" % (record, record_offset)
        )

    def _read_string(self):
        object_id = self._i32()
        value = self._string()
        self.objects[object_id] = value
        return value

    def _read_primitive_array(self):
        object_id = self._i32()
        length = self._i32()
        primitive_type = self._u8()
        value = self._primitive_values(length, primitive_type)
        self.objects[object_id] = value
        return value

    def _primitive_values(self, length, primitive_type):
        if length < 0:
            raise self._error("negative array length")
        dtype = _NUMPY_PRIMITIVES.get(primitive_type)
        if dtype is None:
            return np.asarray(
                [self._primitive(primitive_type) for unused in range(length)]
            )
        dtype = np.dtype(dtype)
        raw = self._read(length * dtype.itemsize)
        return np.frombuffer(raw, dtype=dtype, count=length)

    def _read_object_array(self):
        object_id = self._i32()
        length = self._i32()
        value = self._read_array_values(length)
        self.objects[object_id] = value
        return value

    def _read_array_values(self, length, discard=False):
        if length < 0:
            raise self._error("negative array length")
        value = []
        count = 0
        while count < length:
            record_offset = self.offset
            record = self._u8()
            if record == _RECORD_NULL:
                null_count = 1
            elif record == _RECORD_NULL_256:
                null_count = self._u8()
            elif record == _RECORD_NULL_MULTIPLE:
                null_count = self._i32()
            else:
                self.offset = record_offset
                item = self._read_inline()
                if not discard:
                    value.append(item)
                count += 1
                continue
            if null_count < 0 or count + null_count > length:
                raise self._error("invalid null run in array")
            if not discard:
                value.extend([None] * null_count)
            count += null_count
        return None if discard else value

    def _read_binary_array(self):
        object_id = self._i32()
        array_type = self._u8()
        rank = self._i32()
        if rank < 1 or rank > 32:
            raise self._error("invalid array rank %d" % rank)
        lengths = [self._i32() for unused in range(rank)]
        if any(length < 0 for length in lengths):
            raise self._error("negative array length")
        if array_type in _ARRAY_OFFSET_TYPES:
            for unused in range(rank):
                self._i32()
        binary_type = self._u8()
        additional = None
        if binary_type in (_TYPE_PRIMITIVE, _TYPE_PRIMITIVE_ARRAY):
            additional = self._u8()
        elif binary_type == _TYPE_SYSTEM_CLASS:
            additional = self._string()
        elif binary_type == _TYPE_CLASS:
            additional = (self._string(), self._i32())
        elif binary_type not in (1, 2, 5, 6):
            raise self._error("unsupported array binary type %d" % binary_type)

        length = 1
        for dimension in lengths:
            length *= dimension
        if binary_type == _TYPE_PRIMITIVE:
            value = self._primitive_values(length, additional)
        else:
            class_name = additional[0] if binary_type == _TYPE_CLASS else additional
            value = self._read_array_values(
                length, discard=class_name in _DISCARDED_ARRAY_CLASSES
            )
        self.objects[object_id] = value
        return value
