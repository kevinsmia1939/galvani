"""Tests for Metrohm AUTOLAB NOVA .nox files."""

# SPDX-FileCopyrightText: 2026 OpenAI
# SPDX-License-Identifier: GPL-3.0-or-later

import io
import struct

import numpy as np
from numpy.testing import assert_allclose
import pytest

from galvani import NOXfile
from galvani.Nova import NRBFError


def _i32(value):
    return struct.pack("<i", value)


def _string(value):
    raw = value.encode("utf-8")
    length = len(raw)
    prefix = bytearray()
    while length >= 0x80:
        prefix.append((length & 0x7f) | 0x80)
        length >>= 7
    prefix.append(length)
    return bytes(prefix) + raw


def _object_string(object_id, value):
    return b"\x06" + _i32(object_id) + _string(value)


def _class(object_id, name, members, types, additional, values):
    record = b"\x05" + _i32(object_id) + _string(name) + _i32(len(members))
    record += b"".join(_string(member) for member in members)
    record += bytes(types) + additional + _i32(1)
    return record + values


def _double_list(object_id, array_id, values, metadata_id=None):
    array = np.asarray(values, dtype="<f8")
    array_record = b"\x0f" + _i32(array_id) + _i32(len(array)) + b"\x06"
    array_record += array.tobytes()
    values_record = array_record + _i32(len(array)) + _i32(0)
    if metadata_id is not None:
        return b"\x01" + _i32(object_id) + _i32(metadata_id) + values_record
    return _class(
        object_id,
        "System.Collections.Generic.List`1[[System.Double, mscorlib]]",
        ("_items", "_size", "_version"),
        (7, 0, 0),
        b"\x06\x08\x08",
        values_record,
    )


def _minimal_nox():
    header = b"\x00" + _i32(1) + _i32(-1) + _i32(1) + _i32(0)
    root = _class(
        1,
        "EcoChemie.Shared.ProcedureDescription",
        (
            "_name",
            "_text",
            "_instrument",
            "_remarks",
            "_timeStamp",
            "_modifiedTimeStamp",
        ),
        (1, 1, 1, 1, 0, 0),
        b"\x0d\x0d",
        _object_string(2, "test")
        + _object_string(3, "synthetic NOVA file")
        + _object_string(4, "AUT123")
        + _object_string(5, "")
        + struct.pack("<QQ", 0, 0),
    )

    parent = _class(
        50,
        "EcoChemie.Autolab.FunctionHandlers.FHLevel",
        ("FunctionHandler+_name", "FunctionHandler+_timeStampEmbedded2"),
        (1, 0),
        b"\x08",
        _object_string(51, "FHLevelGalvanostatic") + _i32(100),
    )
    parameter_members = (
        "_parameter",
        "CommandParameter+_name",
        "CommandParameter+_unit",
        "CommandParameter+_parent",
        "CommandParameter+_text",
    )
    parameter_types = (2, 1, 1, 2, 1)

    first_values = (
        _double_list(30, 31, [0.0, 1.0, 2.0])
        + _object_string(32, "CalcTime")
        + _object_string(33, "s")
        + parent
        + _object_string(34, "Time")
    )
    time_parameter = _class(
        20,
        "EcoChemie.Utils.Sequencer.CommandParameterDataArray",
        parameter_members,
        parameter_types,
        b"",
        first_values,
    )

    def parameter(object_id, list_id, array_id, name_id, name, unit, data):
        values = (
            _double_list(list_id, array_id, data, metadata_id=30)
            + _object_string(name_id, name)
            + _object_string(name_id + 1, unit)
            + b"\x09"
            + _i32(50)
            + _object_string(name_id + 2, name)
        )
        return b"\x01" + _i32(object_id) + _i32(20) + values

    potential = parameter(
        21, 40, 41, 42, "EI_0.CalcPotential", "V", [3.0, 3.1, 3.2]
    )
    current = parameter(
        22, 45, 46, 47, "EI_0.CalcCurrent", "A", [0.1, 0.2, 0.3]
    )
    return header + root + time_parameter + potential + current + b"\x0b"


def test_read_minimal_nox():
    nox = NOXfile(io.BytesIO(_minimal_nox()))

    assert nox.name == "test"
    assert nox.description == "synthetic NOVA file"
    assert nox.instrument == "AUT123"
    assert nox.stream_count == 1
    assert nox.data.dtype.names == ("time/s", "Ewe/V", "I/A", "index", "segment")
    assert_allclose(nox.data["time/s"], [0.0, 1.0, 2.0])
    assert_allclose(nox.data["Ewe/V"], [3.0, 3.1, 3.2])
    assert_allclose(nox.data["I/A"], [0.1, 0.2, 0.3])
    assert nox.datasets[0].units["EI_0.CalcCurrent"] == "A"


def test_reject_non_nox_file():
    with pytest.raises(NRBFError, match="NRBF header"):
        NOXfile(io.BytesIO(b"not a NOVA file"))
