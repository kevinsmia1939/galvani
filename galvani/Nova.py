# -*- coding: utf-8 -*-
"""Read Metrohm AUTOLAB NOVA ``.nox`` files.

NOVA files contain consecutive .NET BinaryFormatter (NRBF) object graphs.  The
reader below treats those graphs as data only; it neither executes serialized
code nor requires NOVA, pythonnet, or the Metrohm SDK.
"""

# SPDX-FileCopyrightText: 2026 OpenAI
# SPDX-License-Identifier: GPL-3.0-or-later

from collections import OrderedDict
from datetime import datetime, timedelta
import hashlib
import os

import numpy as np

from ._nrbf import NRBFError, NRBFParser


__all__ = ["NOXfile", "NovaDataset", "NRBFError"]


_PARAMETER_DATA_ARRAY = "EcoChemie.Utils.Sequencer.CommandParameterDataArray"
_GENERIC_LIST = "System.Collections.Generic.List`1"


def _string(parser, value, default=None):
    value = parser.resolve(value)
    return value if isinstance(value, str) else default


def _datetime_from_binary(value):
    """Convert the ticks portion of a System.DateTime binary value."""
    if not isinstance(value, (int, np.integer)):
        return None
    ticks = int(value) & 0x3fffffffffffffff
    try:
        return datetime(1, 1, 1) + timedelta(microseconds=ticks // 10)
    except (OverflowError, ValueError):
        return None


def _numeric_list(parser, value, seen=None):
    """Find the numeric List<T> wrapped by a NOVA parameter object."""
    if seen is None:
        seen = set()

    if isinstance(value, dict) and set(value) == {"__ref__"}:
        object_id = value["__ref__"]
        if object_id in seen:
            return None
        seen.add(object_id)
        return _numeric_list(parser, parser.objects.get(object_id), seen)

    value = parser.resolve(value)
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return None
        seen.add(marker)
        class_name = value.get("__class__", "")
        if class_name.startswith(_GENERIC_LIST) and (
            "[[System.Double," in class_name or "[[System.Int32," in class_name
        ):
            items = parser.resolve(value.get("_items"))
            if isinstance(items, np.ndarray):
                size = value.get("_size", len(items))
                if not isinstance(size, int) or size < 0 or size > len(items):
                    raise NRBFError("invalid size in NOVA numeric list")
                return items[:size], id(items)

        # The useful route is normally ParameterObject._value.  Trying it
        # first avoids walking repeated base-class parameter references.
        keys = ["_value", "ParameterObject+_value"]
        keys.extend(key for key in value if key not in keys and key != "__class__")
        for key in keys:
            if key in value:
                result = _numeric_list(parser, value[key], seen)
                if result is not None:
                    return result
    return None


class NovaDataset:
    """A set of aligned signals recorded by one NOVA command.

    Signal names are the names stored by NOVA, for example ``CalcTime`` and
    ``EI_0.CalcPotential``.  Use ``dataset[name]`` or ``dataset.signals`` to
    access their NumPy arrays.
    """

    def __init__(self, command, command_id, timestamp, signals, units, labels):
        self.command = command
        self.command_id = command_id
        self.timestamp = timestamp
        self.signals = signals
        self.units = units
        self.labels = labels

    def __getitem__(self, name):
        return self.signals[name]

    def __contains__(self, name):
        return name in self.signals

    def __len__(self):
        if not self.signals:
            return 0
        return max(len(value) for value in self.signals.values())

    @property
    def signal_names(self):
        return tuple(self.signals)

    def __repr__(self):
        return "NovaDataset(command=%r, npts=%d, signals=%r)" % (
            self.command,
            len(self),
            self.signal_names,
        )


def _extract_datasets(parser):
    groups = OrderedDict()

    for object_id, value in parser.objects.items():
        if (
            not isinstance(value, dict)
            or value.get("__class__") != _PARAMETER_DATA_ARRAY
        ):
            continue
        result = _numeric_list(parser, value.get("_parameter"))
        if result is None:
            continue
        array, source_id = result
        if len(array) == 0:
            continue

        name = _string(parser, value.get("CommandParameter+_name"))
        if not name:
            continue
        parent = parser.resolve(value.get("CommandParameter+_parent"))
        if isinstance(parent, dict):
            parent_marker = id(parent)
            command = _string(parser, parent.get("FunctionHandler+_name"), "")
            timestamp = parent.get("FunctionHandler+_timeStampEmbedded2")
        else:
            parent_marker = object_id
            command = ""
            timestamp = None

        group = groups.setdefault(
            parent_marker,
            {
                "command": command,
                "command_id": object_id,
                "timestamp": timestamp,
                "signals": OrderedDict(),
                "sources": {},
                "units": {},
                "labels": {},
            },
        )
        group["signals"][name] = array
        group["sources"][name] = source_id
        group["units"][name] = _string(
            parser, value.get("CommandParameter+_unit"), ""
        )
        group["labels"][name] = _string(
            parser, value.get("CommandParameter+_text"), name
        )

    # Executed commands and stored repeat templates can point at the same
    # signal lists.  Merge such views instead of exposing duplicates.
    merged = OrderedDict()
    for group in groups.values():
        time_name = "CalcTime" if "CalcTime" in group["sources"] else next(
            iter(group["sources"])
        )
        time_array = group["signals"][time_name]
        digest = hashlib.blake2b(time_array.view(np.uint8), digest_size=16).digest()
        key = (time_name, time_array.dtype.str, len(time_array), digest)
        if key not in merged:
            merged[key] = group
        else:
            previous = merged[key]
            previous["signals"].update(group["signals"])
            previous["sources"].update(group["sources"])
            previous["units"].update(group["units"])
            previous["labels"].update(group["labels"])

    datasets = []
    for group in merged.values():
        # Copy only the useful calculated arrays.  The parser can then release
        # its view of the complete input file.
        signals = OrderedDict(
            (name, np.array(array, copy=True))
            for name, array in group["signals"].items()
        )
        datasets.append(
            NovaDataset(
                command=group["command"],
                command_id=group["command_id"],
                timestamp=group["timestamp"],
                signals=signals,
                units=dict(group["units"]),
                labels=dict(group["labels"]),
            )
        )
    return datasets


def _combined_data(datasets):
    """Create the convenient, Galvani-style table of core cycling signals."""
    required = ("CalcTime", "EI_0.CalcPotential", "EI_0.CalcCurrent")
    selected = [dataset for dataset in datasets if all(x in dataset for x in required)]
    selected.sort(key=lambda dataset: float(dataset["CalcTime"][0]))

    dtype = np.dtype(
        [
            ("time/s", "<f8"),
            ("Ewe/V", "<f8"),
            ("I/A", "<f8"),
            ("index", "<i8"),
            ("segment", "<u4"),
        ]
    )
    sizes = [min(len(dataset[name]) for name in required) for dataset in selected]
    data = np.empty(sum(sizes), dtype=dtype)
    position = 0
    for segment, (dataset, size) in enumerate(zip(selected, sizes), 1):
        target = data[position : position + size]
        target["time/s"] = dataset["CalcTime"][:size]
        target["Ewe/V"] = dataset["EI_0.CalcPotential"][:size]
        target["I/A"] = dataset["EI_0.CalcCurrent"][:size]
        if "Index" in dataset and len(dataset["Index"]) >= size:
            target["index"] = dataset["Index"][:size]
        else:
            target["index"] = np.arange(1, size + 1)
        target["segment"] = segment
        position += size
    return data


class NOXfile:
    """Metrohm AUTOLAB NOVA ``.nox`` file.

    Parameters
    ----------
    file_or_path
        A path-like object or an open binary file.

    Attributes
    ----------
    data
        Structured NumPy array combining all command datasets that contain
        time, potential, and current.  Its fields are ``time/s``, ``Ewe/V``,
        ``I/A``, ``index``, and ``segment``.
    datasets
        Individual :class:`NovaDataset` objects, including partial datasets
        that do not have all three core signals.
    name, description, instrument, timestamp
        Procedure metadata from the final object graph in the file.
    """

    def __init__(self, file_or_path):
        if isinstance(file_or_path, (str, bytes, os.PathLike)):
            with open(file_or_path, "rb") as nox_file:
                raw = nox_file.read()
        else:
            raw = file_or_path.read()
        if not isinstance(raw, bytes):
            raise TypeError(".nox files must be opened in binary mode")

        offset = 0
        roots = []
        parsers = []
        while offset < len(raw):
            parser = NRBFParser(raw, offset)
            roots.append(parser.parse())
            parsers.append(parser)
            if parser.offset <= offset:
                raise NRBFError("NRBF parser made no progress")
            offset = parser.offset
        if not roots or not isinstance(roots[-1], dict):
            raise NRBFError("NOVA file does not contain a procedure description")

        root = roots[-1]
        parser = parsers[-1]
        self.name = _string(parser, root.get("_name"), "")
        self.description = _string(parser, root.get("_text"), "")
        self.instrument = _string(parser, root.get("_instrument"), "")
        self.remarks = _string(parser, root.get("_remarks"), "")
        self.timestamp = _datetime_from_binary(root.get("_timeStamp"))
        self.modified_timestamp = _datetime_from_binary(root.get("_modifiedTimeStamp"))
        self.datasets = _extract_datasets(parser)
        self.data = _combined_data(self.datasets)
        self.dtype = self.data.dtype
        self.npts = len(self.data)
        self.stream_count = len(roots)
