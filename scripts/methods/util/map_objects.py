"""Shared object localization helpers for SemPathBench maps."""

from __future__ import annotations

import builtins
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
# Keep this alias compatible with the Python 3.9 environment used by Lang2LTL.
# ``typing.TypeAlias`` was only added in Python 3.10 and is not needed at
# runtime here.
Cell = tuple[int, int]


def _grid_size(map_state: Mapping[str, object]) -> int:
    raw_size = map_state.get("grid_size")
    if isinstance(raw_size, int) and not isinstance(raw_size, bool) and raw_size > 0:
        return raw_size
    layers = map_state.get("layers")
    occupancy = layers.get("occupancy") if isinstance(layers, Mapping) else None
    return len(occupancy) if isinstance(occupancy, Sequence) else 0


def _layer(map_state: Mapping[str, object], name: str) -> Sequence[Sequence[object]]:
    layers = map_state.get("layers")
    if not isinstance(layers, Mapping):
        return []
    layer = layers.get(name)
    if not isinstance(layer, Sequence) or isinstance(layer, (str, bytes)):
        return []
    return layer  # type: ignore[return-value]


def _value_at(layer: Sequence[Sequence[object]], row: int, col: int) -> int:
    if row < 0 or row >= len(layer):
        return 0
    row_values = layer[row]
    if not isinstance(row_values, Sequence) or isinstance(row_values, (str, bytes)):
        return 0
    if col < 0 or col >= len(row_values):
        return 0
    try:
        return int(row_values[col])
    except (TypeError, ValueError):
        return 0


def _valid_cell(value: object, grid_size: int) -> Cell | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        return None
    row, col = value
    if not isinstance(row, (int, float)) or not isinstance(col, (int, float)):
        return None
    cell = (int(round(float(row))), int(round(float(col))))
    if 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size:
        return cell
    return None


def _valid_cells(value: object, grid_size: int) -> list[Cell]:
    if not builtins.isinstance(value, Sequence) or builtins.isinstance(value, (str, bytes)):
        return []
    cells: list[Cell] = []
    seen_cells: set[Cell] = builtins.set()
    for item in value:
        cell = _valid_cell(item, grid_size)
        if cell is None or cell in seen_cells:
            continue
        cells.append(cell)
        seen_cells.add(cell)
    return cells


def _metadata_by_id(items: object) -> dict[int, Mapping[str, object]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return {}
    result: dict[int, Mapping[str, object]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        raw_id = item.get("id")
        if isinstance(raw_id, int) and not isinstance(raw_id, bool) and raw_id > 0:
            result[raw_id] = item
    return result


def object_instances_by_id(map_state: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    """Return object metadata records keyed by SemPathBench object id."""

    return _metadata_by_id(map_state.get("object_instances"))


def object_center_hint(
    map_state: Mapping[str, object],
    object_id: int,
) -> Cell | None:
    """Return a one-cell center fallback for an object when available."""

    grid_size = _grid_size(map_state)
    instance = object_instances_by_id(map_state).get(object_id)
    if not instance:
        return None
    for key in ("center_grid", "metadata_grid_cell"):
        cell = _valid_cell(instance.get(key), grid_size)
        if cell is not None:
            return cell
    return None


def _footprint_cells_by_id(map_state: Mapping[str, object]) -> dict[int, list[Cell]]:
    grid_size = _grid_size(map_state)
    raw_footprints = map_state.get("object_footprints")
    if not isinstance(raw_footprints, Mapping):
        return {}
    result: dict[int, list[Cell]] = {}
    for raw_key, raw_record in raw_footprints.items():
        object_id: int | None = None
        if isinstance(raw_record, Mapping):
            raw_object_id = raw_record.get("object_id")
            if isinstance(raw_object_id, int) and not isinstance(raw_object_id, bool):
                object_id = raw_object_id
        if object_id is None:
            try:
                object_id = int(raw_key)
            except (TypeError, ValueError):
                continue
        if object_id <= 0 or not isinstance(raw_record, Mapping):
            continue
        cells = _valid_cells(raw_record.get("cells"), grid_size)
        if not cells:
            for key in ("center_grid", "metadata_grid_cell"):
                cell = _valid_cell(raw_record.get(key), grid_size)
                if cell is not None:
                    cells = [cell]
                    break
        if cells:
            result[object_id] = cells
    return result


def _legacy_layer_cells_by_id(map_state: Mapping[str, object]) -> dict[int, list[Cell]]:
    object_layer = _layer(map_state, "object_instance")
    result: dict[int, list[Cell]] = defaultdict(list)
    # Use indexed iteration here rather than ``enumerate``. This helper can be
    # called after method-specific dependencies have been imported, and a
    # dependency that replaces the ``enumerate`` builtin can return items that
    # are not ``(index, value)`` pairs. JSON grid rows are indexable sequences,
    # so indexed access is sufficient and insulated from that mutation.
    for row in range(len(object_layer)):
        row_values = object_layer[row]
        if not isinstance(row_values, Sequence) or isinstance(row_values, (str, bytes)):
            continue
        for col in range(len(row_values)):
            raw_value = row_values[col]
            try:
                object_id = int(raw_value)
            except (TypeError, ValueError):
                continue
            if object_id > 0:
                result[object_id].append((row, col))
    return dict(result)


def object_cells_by_id(map_state: Mapping[str, object]) -> dict[int, list[Cell]]:
    """Return localizing cells for every known object.

    Priority is:
    1. explicit ``object_footprints``;
    2. legacy ``layers.object_instance``;
    3. per-object center hints.
    """

    result = _footprint_cells_by_id(map_state)
    for object_id, cells in _legacy_layer_cells_by_id(map_state).items():
        result.setdefault(object_id, cells)
    for object_id in object_instances_by_id(map_state):
        if object_id not in result:
            center = object_center_hint(map_state, object_id)
            if center is not None:
                result[object_id] = [center]
    return {object_id: cells for object_id, cells in sorted(result.items())}


def object_cells(map_state: Mapping[str, object], object_id: int) -> list[Cell]:
    """Return localizing cells for one object id."""

    return object_cells_by_id(map_state).get(object_id, [])


def object_center(
    map_state: Mapping[str, object],
    object_id: int,
) -> tuple[float, float] | None:
    """Return the centroid of an object's localizing cells."""

    cells = object_cells(map_state, object_id)
    if not cells:
        return None
    return (
        sum(row for row, _col in cells) / len(cells),
        sum(col for _row, col in cells) / len(cells),
    )


def cell_object_ids(
    map_state: Mapping[str, object],
    row: int,
    col: int,
) -> tuple[int, ...]:
    """Return all object ids associated with one grid cell."""

    raw_index = map_state.get("cell_object_ids")
    if isinstance(raw_index, Mapping):
        raw_ids = raw_index.get(f"{row},{col}")
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes)):
            ids: list[int] = []
            seen: set[int] = set()
            for raw_id in raw_ids:
                try:
                    object_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if object_id > 0 and object_id not in seen:
                    ids.append(object_id)
                    seen.add(object_id)
            if ids:
                return tuple(ids)

    ids = [
        object_id
        for object_id, cells in object_cells_by_id(map_state).items()
        if (row, col) in cells
    ]
    if ids:
        return tuple(ids)

    legacy_value = _value_at(_layer(map_state, "object_instance"), row, col)
    return (legacy_value,) if legacy_value > 0 else ()


def object_room_id(
    map_state: Mapping[str, object],
    object_id: int,
) -> int | None:
    """Return the room id associated with an object when it can be inferred."""

    instance = object_instances_by_id(map_state).get(object_id)
    if instance:
        raw_room_id = instance.get("room_id")
        if isinstance(raw_room_id, int) and not isinstance(raw_room_id, bool) and raw_room_id > 0:
            return raw_room_id
        if isinstance(raw_room_id, float) and raw_room_id > 0:
            return int(raw_room_id)

    cells = object_cells(map_state, object_id)
    if not cells:
        return None
    room_layer = _layer(map_state, "room")
    votes: Counter[int] = Counter()
    for row, col in cells:
        room_id = _value_at(room_layer, row, col)
        if room_id > 0:
            votes[room_id] += 1
    if votes:
        return votes.most_common(1)[0][0]

    center = object_center(map_state, object_id)
    if center is None:
        return None
    room_id = _value_at(room_layer, int(round(center[0])), int(round(center[1])))
    return room_id if room_id > 0 else None
