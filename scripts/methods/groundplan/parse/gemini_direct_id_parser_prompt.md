# Task

Ground the route instruction directly to canonical IDs from the compact scene
catalog. The catalog is derived from the semantic map and contains no benchmark
constraints, human trajectory, or evaluator information.

Instruction:

```text
{{INSTRUCTION}}
```

Compact scene catalog:

```json
{{SCENE_CATALOG}}
```

# Output schema

Return exactly one JSON object:

```json
{
  "segments": [
    {"id": "s1", "target_id": "object_17"}
  ],
  "constraints": [
    {
      "kind": "forbid",
      "ref_ids": ["object_25"],
      "segment_ids": ["s1"],
      "spatial_scope_ids": ["room_2"],
      "source_text": "avoid the chair while going to the sofa"
    }
  ]
}
```

Allowed constraint kinds are `require_visit`, `require_visit_in_order`,
`forbid`, `prefer_near`, `prefer_far`, `prefer_relative`, and
`prefer_path_shape`.

- Use only IDs that appear verbatim in the catalog.
- `segments` must preserve every destination and its order. The first segment
  starts at `task_start`; later segments start at the previous target.
- Put a constraint in every segment where it is active. If `segment_ids` is
  omitted, it is treated as active in all segments.
- `spatial_scope_ids` is optional and may contain room or object IDs.
- `prefer_relative` needs two `ref_ids` and may set `relation` to `closer_to`
  or `farther_from`.
- `prefer_path_shape` may include `path_shape` with type `through`, `circle`,
  or `follow_wall`, plus optional `direction` and `fraction`.
- Do not invent IDs, coordinates, paths, dense cells, or planner parameters.
- Do not output Markdown, comments, or explanations.

