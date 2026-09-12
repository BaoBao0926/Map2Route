#!/usr/bin/env python3
"""Tests for ProcTHOR scene conversion helpers."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.make_maps.procthor.convert_procthor_scene import (
    build_object_footprint_maps,
    build_room_type_map,
    canonicalize_object_category,
    compute_house_largest_room_size,
    compute_house_scene_size,
    convert_procthor_house_to_maps,
)
from scripts.make_maps.procthor.transform_procthor_to_map import (
    build_default_output_prefix,
    expected_scene_output_paths,
    expected_simple_demo_output_paths,
    main,
    map_split_for_sequence_position,
    map_split_for_source_index,
    parse_args,
    transform_procthor_batch,
    transform_procthor_to_map,
)
from scripts.make_maps.procthor.view_procthor_map import format_room_size_summary


class FakeEvent:
    def __init__(
        self,
        metadata: dict[str, object],
        third_party_camera_frames: list[np.ndarray] | None = None,
    ) -> None:
        self.metadata = metadata
        self.third_party_camera_frames = third_party_camera_frames or []


class FakeController:
    last_init_kwargs: dict[str, object] = {}
    last_step_kwargs: dict[str, dict[str, object]] = {}

    def __init__(self, scene: dict[str, object], **_kwargs) -> None:
        del scene
        type(self).last_init_kwargs = dict(_kwargs)
        type(self).last_step_kwargs = {}
        self.stopped = False
        self.last_event = FakeEvent(
            {
                "objects": [
                    {
                        "objectId": "Floor|runtime",
                        "objectType": "Floor",
                        "name": "Floor",
                        "position": {"x": 2.0, "y": 0.0, "z": 2.0},
                    },
                    {
                        "objectId": "Wall|runtime",
                        "objectType": "Wall",
                        "name": "Wall",
                        "position": {"x": 2.0, "y": 0.0, "z": 2.0},
                    },
                    {
                        "objectId": "Table|runtime",
                        "objectType": "Table",
                        "name": "Table",
                        "position": {"x": 2.0, "y": 0.0, "z": 2.0},
                        "axisAlignedBoundingBox": {
                            "cornerPoints": [
                                [1.5, 0.0, 1.5],
                                [1.5, 0.0, 2.5],
                                [2.5, 0.0, 1.5],
                                [2.5, 0.0, 2.5],
                            ]
                        },
                    }
                ]
            }
        )

    def step(self, action: str, **_kwargs) -> FakeEvent:
        type(self).last_step_kwargs[action] = dict(_kwargs)
        if action == "GetReachablePositions":
            return FakeEvent(
                {
                    "actionReturn": [
                        {"x": 0.0, "z": 0.0},
                        {"x": 4.0, "z": 0.0},
                        {"x": 0.0, "z": 4.0},
                        {"x": 4.0, "z": 4.0},
                    ]
                }
            )
        if action == "GetMapViewCameraProperties":
            return FakeEvent({"actionReturn": {}})
        if action == "AddThirdPartyCamera":
            return FakeEvent({}, [np.zeros((8, 8, 3), dtype=np.uint8)])
        raise AssertionError(f"Unexpected fake controller action: {action}")

    def stop(self) -> None:
        self.stopped = True


class MapSplitTest(unittest.TestCase):
    def test_map_split_assigns_two_train_then_one_valunseen(self) -> None:
        self.assertEqual(
            [map_split_for_source_index(index) for index in range(6)],
            [
                "train",
                "train",
                "valunseen",
                "train",
                "train",
                "valunseen",
            ],
        )

    def test_sequence_position_split_is_one_based(self) -> None:
        self.assertEqual(
            [map_split_for_sequence_position(position) for position in range(1, 7)],
            [
                "train",
                "train",
                "valunseen",
                "train",
                "train",
                "valunseen",
            ],
        )


def make_house(width: float = 4.0, depth: float = 4.0) -> dict[str, object]:
    return {
        "rooms": [
            {
                "id": "room|0",
                "roomType": "LivingRoom",
                "floorPolygon": [
                    {"x": 0.0, "z": 0.0},
                    {"x": 0.0, "z": depth},
                    {"x": width, "z": depth},
                    {"x": width, "z": 0.0},
                ],
            }
        ],
        "objects": [
            {
                "id": "Table|0",
                "assetId": "table_asset",
                "position": {"x": 2.0, "y": 0.0, "z": 2.0},
            },
            {
                "id": "Floor|0",
                "assetId": "floor_asset",
                "position": {"x": 1.0, "y": 0.0, "z": 1.0},
            },
            {
                "id": "Wall|0",
                "assetId": "wall_asset",
                "position": {"x": 3.0, "y": 0.0, "z": 3.0},
            }
        ],
    }


def make_house_with_rooms(room_specs: list[tuple[float, float]]) -> dict[str, object]:
    house = make_house()
    rooms: list[dict[str, object]] = []
    x_offset = 0.0
    for room_index, (width, depth) in enumerate(room_specs):
        rooms.append(
            {
                "id": f"room|{room_index}",
                "roomType": "LivingRoom",
                "floorPolygon": [
                    {"x": x_offset, "z": 0.0},
                    {"x": x_offset, "z": depth},
                    {"x": x_offset + width, "z": depth},
                    {"x": x_offset + width, "z": 0.0},
                ],
            }
        )
        x_offset += width + 1.0
    house["rooms"] = rooms
    return house


def write_minimal_complete_export(output_prefix: Path) -> None:
    maps_path, metadata_path, _overview_path = expected_scene_output_paths(output_prefix)
    simple_json_path, _simple_png_path, simple_ppm_path, template_path, thinggraph_path = (
        expected_simple_demo_output_paths(output_prefix)
    )
    maps_path.parent.mkdir(parents=True, exist_ok=True)
    maps_path.write_bytes(b"existing maps")
    metadata_path.write_text("{}", encoding="utf-8")
    simple_json_path.write_text("{}", encoding="utf-8")
    simple_ppm_path.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
    template_path.write_text("{}", encoding="utf-8")
    thinggraph_path.write_text("{}", encoding="utf-8")


class RoomTypeMapTest(unittest.TestCase):
    def test_default_export_path_is_grouped_by_map_split(self) -> None:
        prefix = build_default_output_prefix("train", 2)
        self.assertTrue(
            prefix.as_posix().endswith(
                "resources/maps/procthor/valunseen/003_valunseen/003_valunseen"
            )
        )

    def test_compute_house_scene_size_sums_room_polygon_areas(self) -> None:
        self.assertAlmostEqual(compute_house_scene_size(make_house()), 16.0)
        self.assertAlmostEqual(compute_house_largest_room_size(make_house()), 16.0)

    def test_room_type_covers_non_traversable_cells_inside_room_polygon(self) -> None:
        house = make_house()
        map_info = {
            "x_min": 0.0,
            "x_max": 5.0,
            "z_min": 0.0,
            "z_max": 5.0,
            "resolution": 1.0,
            "H": 6,
            "W": 6,
        }
        traversibility_map = np.zeros((6, 6), dtype=np.uint8)
        traversibility_map[1, 1] = 1

        room_type_map, room_type_to_id, _room_id_to_type = build_room_type_map(
            house,
            map_info,
            traversibility_map=traversibility_map,
        )

        living_room_id = room_type_to_id["LivingRoom"]
        self.assertEqual(room_type_map[1, 1], living_room_id)
        self.assertEqual(room_type_map[2, 2], living_room_id)
        self.assertEqual(room_type_map[4, 4], living_room_id)
        self.assertEqual(room_type_map[5, 5], -1)

    def test_convert_house_to_maps_without_controller(self) -> None:
        scene = convert_procthor_house_to_maps(make_house(), resolution=1.0, padding=0.0)

        self.assertEqual(scene["traversibility_map"].shape, (5, 5))
        self.assertEqual(scene["room_type_map"].shape, (5, 5))
        self.assertEqual(scene["object_category_map"].shape, (5, 5))
        self.assertEqual(scene["object_instance_map"].shape, (5, 5))
        self.assertEqual(scene["room_type_to_id"], {"LivingRoom": 0})
        self.assertEqual(scene["category_to_id"], {"Table": 1, "Wall": 2})
        self.assertEqual(scene["object_category_map"][2, 2], 1)
        self.assertEqual(scene["object_category_map"][3, 3], 2)
        self.assertEqual(scene["room_type_map"][2, 2], 0)
        self.assertEqual(len(scene["room_metadata"]), 1)
        room_record = scene["room_metadata"][0]
        self.assertEqual(room_record["room_id"], "room|0")
        self.assertEqual(room_record["room_type"], "LivingRoom")
        self.assertAlmostEqual(room_record["area"], 16.0)
        self.assertEqual(room_record["num_grid_cells"], 25)
        self.assertAlmostEqual(room_record["grid_area"], 25.0)
        self.assertEqual(room_record["bbox"]["width_x"], 4.0)
        self.assertEqual(room_record["bbox"]["depth_z"], 4.0)
        self.assertEqual(room_record["centroid_xz"], [2.0, 2.0])

    def test_room_size_summary_formats_true_area_for_visualization(self) -> None:
        scene = convert_procthor_house_to_maps(make_house(), resolution=1.0, padding=0.0)

        summary = format_room_size_summary(scene["room_metadata"])

        self.assertEqual(summary, "room sizes: LivingRoom=16.00 (4.00x4.00)")

    def test_convert_house_to_maps_can_use_runtime_object_footprints(self) -> None:
        scene = convert_procthor_house_to_maps(
            make_house(),
            controller=FakeController(make_house()),
            resolution=1.0,
            padding=0.0,
        )

        self.assertEqual(scene["category_to_id"], {"Table": 1, "Wall": 2})
        self.assertEqual(scene["object_category_map"][2, 2], 1)
        table_record = next(
            record
            for record in scene["object_metadata"]
            if record["objectType"] == "Table"
        )
        self.assertEqual(table_record["footprint_source"], "bbox")

    def test_wall_footprint_uses_object_id_segment_instead_of_bbox(self) -> None:
        controller = FakeController(make_house())
        controller.last_event = FakeEvent(
            {
                "objects": [
                    {
                        "objectId": "wall|0|0.0|2.0|4.0|2.0",
                        "objectType": "Wall",
                        "name": "Wall",
                        "position": {"x": 2.0, "y": 0.0, "z": 2.0},
                        "axisAlignedBoundingBox": {
                            "cornerPoints": [
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 4.0],
                                [4.0, 0.0, 0.0],
                                [4.0, 0.0, 4.0],
                            ]
                        },
                    }
                ]
            }
        )
        map_info = {
            "x_min": 0.0,
            "x_max": 4.0,
            "z_min": 0.0,
            "z_max": 4.0,
            "resolution": 1.0,
            "H": 5,
            "W": 5,
        }

        object_category_map, _instance_map, metadata, category_to_id, _ = (
            build_object_footprint_maps(controller, map_info)
        )

        self.assertEqual(category_to_id, {"Wall": 1})
        self.assertEqual(int(np.count_nonzero(object_category_map)), 5)
        self.assertTrue(np.all(object_category_map[2, :] == 1))
        self.assertEqual(metadata[0]["footprint_source"], "wall_segment")

    def test_shelving_unit_is_canonicalized_to_shelf(self) -> None:
        controller = FakeController(make_house())
        controller.last_event = FakeEvent(
            {
                "objects": [
                    {
                        "objectId": "Shelf|0",
                        "objectType": "Shelf",
                        "name": "Shelf",
                        "position": {"x": 1.0, "y": 0.0, "z": 1.0},
                    },
                    {
                        "objectId": "ShelvingUnit|0",
                        "objectType": "ShelvingUnit",
                        "name": "ShelvingUnit",
                        "position": {"x": 3.0, "y": 0.0, "z": 3.0},
                    },
                ]
            }
        )
        map_info = {
            "x_min": 0.0,
            "x_max": 4.0,
            "z_min": 0.0,
            "z_max": 4.0,
            "resolution": 1.0,
            "H": 5,
            "W": 5,
        }

        _category_map, _instance_map, object_metadata, category_to_id, _id_to_category = (
            build_object_footprint_maps(controller, map_info)
        )

        self.assertEqual(canonicalize_object_category("ShelvingUnit"), "Shelf")
        self.assertEqual(category_to_id, {"Shelf": 1})
        self.assertEqual({record["objectType"] for record in object_metadata}, {"Shelf"})
        alias_record = next(
            record for record in object_metadata if record["objectId"] == "ShelvingUnit|0"
        )
        self.assertEqual(alias_record["raw_objectType"], "ShelvingUnit")

    def test_transform_entrypoint_uses_dataset_without_simulator(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_prefix = Path(tmp_dir) / "train_00000"
            _scene, maps_path, metadata_path, overview_path = transform_procthor_to_map(
                split="train",
                index=0,
                dataset={"train": [make_house()]},
                output_prefix=output_prefix,
                resolution=1.0,
                padding=0.0,
                save_overview=False,
                use_simulator=False,
            )

            self.assertTrue(maps_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertIsNone(overview_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["room_metadata"][0]["area"], 16.0)

    def test_transform_entrypoint_can_use_simulator_controller(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_prefix = Path(tmp_dir) / "train_00000"
            _scene, maps_path, metadata_path, overview_path = transform_procthor_to_map(
                split="train",
                index=0,
                dataset={"train": [make_house()]},
                output_prefix=output_prefix,
                resolution=1.0,
                padding=0.0,
                save_overview=False,
                ai2thor_base_dir=None,
                controller_class=FakeController,
            )

            self.assertTrue(maps_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertIsNone(overview_path)
            self.assertEqual(FakeController.last_init_kwargs["gridSize"], 1.0)
            self.assertEqual(
                FakeController.last_step_kwargs["GetReachablePositions"]["gridSize"],
                1.0,
            )

    def test_transform_resolution_overrides_controller_grid_size(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_prefix = Path(tmp_dir) / "train_00000"
            transform_procthor_to_map(
                split="train",
                index=0,
                dataset={"train": [make_house()]},
                output_prefix=output_prefix,
                resolution=1.0,
                padding=0.0,
                save_overview=False,
                ai2thor_base_dir=None,
                controller_class=FakeController,
                controller_kwargs={"gridSize": 0.25},
            )

            self.assertEqual(FakeController.last_init_kwargs["gridSize"], 1.0)
            self.assertEqual(
                FakeController.last_step_kwargs["GetReachablePositions"]["gridSize"],
                1.0,
            )

    def test_transform_entrypoint_saves_topdown_overview_by_default(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_prefix = Path(tmp_dir) / "train_00000"
            _scene, _maps_path, _metadata_path, overview_path = transform_procthor_to_map(
                split="train",
                index=0,
                dataset={"train": [make_house()]},
                output_prefix=output_prefix,
                resolution=1.0,
                padding=0.0,
                ai2thor_base_dir=None,
                controller_class=FakeController,
            )

            self.assertIsNotNone(overview_path)
            self.assertTrue(overview_path.exists())

    def test_transform_entrypoint_saves_simple_demo_style_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_prefix = Path(tmp_dir) / "train_00000"
            transform_procthor_to_map(
                split="train",
                index=0,
                dataset={"train": [make_house()]},
                output_prefix=output_prefix,
                resolution=1.0,
                padding=0.0,
                save_overview=False,
                ai2thor_base_dir=None,
                controller_class=FakeController,
            )

            (
                simple_json_path,
                simple_png_path,
                simple_ppm_path,
                template_path,
                thinggraph_path,
            ) = (
                expected_simple_demo_output_paths(output_prefix)
            )
            self.assertTrue(simple_json_path.exists())
            self.assertTrue(simple_png_path.exists())
            self.assertTrue(simple_ppm_path.exists())
            self.assertTrue(template_path.exists())
            self.assertTrue(thinggraph_path.exists())

            _maps_path, metadata_path, _overview_path = expected_scene_output_paths(
                output_prefix
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["map_id"], "001_train")
            self.assertEqual(metadata["map_split"], "train")
            self.assertEqual(metadata["map_index"], 1)
            self.assertEqual(metadata["procthor_split"], "train")
            self.assertEqual(metadata["procthor_index"], 0)
            self.assertEqual(metadata["procthor_scene_id"], "train_00000")

            payload = json.loads(simple_json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["grid_size"], 5)
            self.assertIn("layer_legends", payload)
            self.assertIn("occupancy", payload["layers"])
            self.assertIn("room", payload["layers"])
            self.assertIn("object_instance", payload["layers"])
            self.assertEqual(payload["room_instances"][0]["category"], "living_room")
            self.assertEqual(payload["metadata"]["map_id"], "001_train")
            self.assertEqual(payload["metadata"]["map_split"], "train")
            self.assertEqual(payload["metadata"]["procthor_scene_id"], "train_00000")
            object_categories = {
                instance["category"] for instance in payload["object_instances"]
            }
            self.assertIn("table", object_categories)
            self.assertIn("wall", object_categories)
            self.assertEqual(payload["metadata"]["original_grid_shape"], [5, 5])

            thinggraph = json.loads(thinggraph_path.read_text(encoding="utf-8"))
            self.assertEqual(thinggraph["graph_type"], "room_object_thinggraph")
            self.assertEqual(thinggraph["map_id"], "001_train")
            self.assertEqual(thinggraph["map_split"], "train")
            self.assertEqual(thinggraph["rooms"][0]["procthor_room_id"], "room|0")
            room_objects = thinggraph["rooms"][0]["objects"]
            self.assertTrue(any(item["category"] == "table" for item in room_objects))
            self.assertTrue(
                all("procthor_object_id" in item for item in room_objects)
            )

    def test_batch_transform_skips_small_scenes_without_counting_them(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            results = transform_procthor_batch(
                split="train",
                number=2,
                scene_size=10.0,
                dataset={
                    "train": [
                        make_house(width=1.0, depth=1.0),
                        make_house(),
                        make_house(),
                    ]
                },
                output_prefix=Path(tmp_dir),
                resolution=1.0,
                padding=0.0,
                save_overview=False,
                ai2thor_base_dir=None,
                controller_class=FakeController,
            )

            self.assertEqual([record["index"] for record in results["skipped"]], [0])
            self.assertEqual([record["index"] for record in results["processed"]], [1, 2])
            self.assertTrue(Path(results["processed"][0]["maps_path"]).exists())
            self.assertIn("001_train", str(results["processed"][0]["maps_path"]))

    def test_batch_transform_assigns_split_by_accepted_position(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            results = transform_procthor_batch(
                split="train",
                number=3,
                scene_size=10.0,
                dataset={
                    "train": [
                        make_house(width=1.0, depth=1.0),
                        make_house(),
                        make_house(),
                        make_house(),
                    ]
                },
                output_prefix=Path(tmp_dir),
                resolution=1.0,
                padding=0.0,
                save_overview=False,
                ai2thor_base_dir=None,
                controller_class=FakeController,
            )

            self.assertEqual([record["index"] for record in results["processed"]], [1, 2, 3])
            self.assertEqual(
                [record["map_split"] for record in results["processed"]],
                ["train", "train", "valunseen"],
            )

            metadata = [
                json.loads(Path(record["metadata_path"]).read_text(encoding="utf-8"))
                for record in results["processed"]
            ]
            self.assertEqual(
                [record["map_index"] for record in metadata],
                [1, 2, 3],
            )
            self.assertEqual(
                [record["map_split"] for record in metadata],
                ["train", "train", "valunseen"],
            )

    def test_batch_transform_bigscene_visits_largest_scenes_first(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            results = transform_procthor_batch(
                split="train",
                number=2,
                sequence="bigscene",
                dataset={
                    "train": [
                        make_house_with_rooms([(10.0, 1.0)]),
                        make_house_with_rooms([(4.0, 4.0), (4.0, 4.0)]),
                        make_house_with_rooms([(6.0, 3.0)]),
                    ]
                },
                output_prefix=Path(tmp_dir),
                resolution=1.0,
                padding=0.0,
                save_overview=False,
                ai2thor_base_dir=None,
                controller_class=FakeController,
            )

            self.assertEqual([record["index"] for record in results["processed"]], [1, 2])
            self.assertEqual(
                [record["scene_size"] for record in results["processed"]],
                [32.0, 18.0],
            )

    def test_batch_transform_bigscene_uses_largest_room_as_tie_breaker(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            results = transform_procthor_batch(
                split="train",
                number=2,
                sequence="bigscene",
                dataset={
                    "train": [
                        make_house_with_rooms([(4.0, 4.0), (2.0, 2.0)]),
                        make_house_with_rooms([(5.0, 4.0)]),
                    ]
                },
                output_prefix=Path(tmp_dir),
                resolution=1.0,
                padding=0.0,
                save_overview=False,
                ai2thor_base_dir=None,
                controller_class=FakeController,
            )

            self.assertEqual([record["index"] for record in results["processed"]], [1, 0])
            self.assertEqual(
                [record["scene_size"] for record in results["processed"]],
                [20.0, 20.0],
            )
            self.assertEqual(
                [record["largest_room_size"] for record in results["processed"]],
                [20.0, 16.0],
            )

    def test_batch_transform_resume_skip_counts_toward_number(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            existing_prefix = Path(tmp_dir) / "001_train" / "001_train"
            write_minimal_complete_export(existing_prefix)

            results = transform_procthor_batch(
                split="train",
                number=1,
                resume=True,
                dataset={"train": [make_house(), make_house()]},
                output_prefix=Path(tmp_dir),
                resolution=1.0,
                padding=0.0,
                save_overview=False,
                ai2thor_base_dir=None,
                controller_class=FakeController,
            )

            self.assertEqual([record["index"] for record in results["skipped"]], [0])
            self.assertEqual(results["skipped"][0]["reason"], "resume")
            self.assertEqual(results["processed"], [])

            _maps_path, metadata_path, _overview_path = expected_scene_output_paths(
                existing_prefix
            )
            simple_json_path, _simple_png_path, _simple_ppm_path, _template_path, thinggraph_path = (
                expected_simple_demo_output_paths(existing_prefix)
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            simple_payload = json.loads(simple_json_path.read_text(encoding="utf-8"))
            thinggraph = json.loads(thinggraph_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["map_id"], "001_train")
            self.assertEqual(metadata["map_split"], "train")
            self.assertEqual(metadata["map_index"], 1)
            self.assertEqual(metadata["procthor_scene_id"], "train_00000")
            self.assertEqual(simple_payload["metadata"]["map_split"], "train")
            self.assertEqual(thinggraph["map_split"], "train")

    def test_parse_args_supports_batch_and_rejects_no_simulator(self) -> None:
        args = parse_args(
            [
                "--index",
                "-1",
                "--number",
                "3",
                "--scene_size",
                "12.5",
                "--sequence",
                "bigscene",
                "--resumet",
                "true",
            ]
        )
        self.assertEqual(args.index, -1)
        self.assertEqual(args.number, 3)
        self.assertEqual(args.scene_size, 12.5)
        self.assertEqual(args.sequence, "bigscene")
        self.assertTrue(args.resume)
        self.assertTrue(parse_args(["--resume"]).resume)
        self.assertFalse(parse_args(["--resume", "false"]).resume)
        self.assertEqual(parse_args(["--sequence", "bigroom"]).sequence, "bigscene")
        self.assertEqual(parse_args(["--sequence", "bvigroom"]).sequence, "bigscene")

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--no-simulator"])

    def test_batch_main_does_not_print_each_skipped_scene(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(
                    [
                        "--index",
                        "-1",
                        "--number",
                        "1",
                        "--scene_size",
                        "999.0",
                        "--output-prefix",
                        tmp_dir,
                        "--skip-overview",
                    ],
                    dataset={
                        "train": [
                            make_house(width=1.0, depth=1.0),
                            make_house(width=2.0, depth=2.0),
                        ]
                    },
                    controller_class=FakeController,
                )

            output = stdout.getvalue()
            self.assertNotIn("Skip train_", output)
            self.assertIn("skipped 2 scenes", output)

    def test_batch_main_prints_progress_only_for_processed_scenes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(
                    [
                        "--index",
                        "-1",
                        "--number",
                        "1",
                        "--scene_size",
                        "10.0",
                        "--output-prefix",
                        tmp_dir,
                        "--skip-overview",
                    ],
                    dataset={
                        "train": [
                            make_house(width=1.0, depth=1.0),
                            make_house(),
                        ]
                    },
                    controller_class=FakeController,
                )

            output = stdout.getvalue()
            self.assertIn("Starting batch: split=train", output)
            self.assertIn("Processing 001_train (1/1", output)
            self.assertIn("procthor_scene_id=train_00001", output)
            self.assertIn("Saved 001_train ->", output)
            self.assertNotIn("train_00000", output)
            self.assertNotIn("Skip train_", output)

    def test_batch_main_prints_resume_skips_but_not_scene_size_skips(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            existing_prefix = Path(tmp_dir) / "001_train" / "001_train"
            write_minimal_complete_export(existing_prefix)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(
                    [
                        "--index",
                        "-1",
                        "--number",
                        "2",
                        "--scene_size",
                        "10.0",
                        "--resume",
                        "true",
                        "--output-prefix",
                        tmp_dir,
                        "--skip-overview",
                    ],
                    dataset={
                        "train": [
                            make_house(width=1.0, depth=1.0),
                            make_house(),
                            make_house(),
                        ]
                    },
                    controller_class=FakeController,
                )

            output = stdout.getvalue()
            self.assertIn("Skip 001_train (1/2, already exists)", output)
            self.assertIn("procthor_scene_id=train_00001", output)
            self.assertIn("Processing 002_train (2/2", output)
            self.assertIn("Saved 002_train ->", output)
            self.assertNotIn("train_00000", output)

    def test_single_scene_main_resume_skips_existing_scene(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_prefix = Path(tmp_dir) / "train_00000"
            write_minimal_complete_export(output_prefix)
            FakeController.last_init_kwargs = {"should": "not change"}

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(
                    [
                        "--index",
                        "0",
                        "--output-prefix",
                        str(output_prefix),
                        "--skip-overview",
                        "--resume",
                        "true",
                    ],
                    dataset={"train": [make_house()]},
                    controller_class=FakeController,
                )

            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(FakeController.last_init_kwargs, {"should": "not change"})


if __name__ == "__main__":
    unittest.main()
