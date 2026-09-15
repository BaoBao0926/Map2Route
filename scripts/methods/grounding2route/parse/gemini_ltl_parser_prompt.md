# Task

Translate the route instruction into a restricted LTL hard-task formula over
propositions grounded to IDs from the compact scene catalog. Quantitative soft
preferences are typed proposition annotations because standard Boolean LTL
does not define preference costs.

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
  "formula": "F(v1 & F(v2)) & G(!a1)",
  "propositions": [
    {"name": "v1", "kind": "visit", "ref_ids": ["object_17"]},
    {"name": "v2", "kind": "visit", "ref_ids": ["room_2"]},
    {"name": "a1", "kind": "avoid", "ref_ids": ["object_25"]},
    {
      "name": "p1",
      "kind": "prefer_near",
      "ref_ids": ["object_31"],
      "during": ["v2"],
      "spatial_scope_ids": ["room_2"]
    }
  ]
}
```

Restricted LTL operators are `F`, `G`, `X`, `!`, `&`, `|`, and parentheses.

- Encode ordered destinations as one nested eventuality chain, for example
  `F(v1 & F(v2 & F(v3)))`.
- A global avoidance must be written as `G(!a1)`.
- A scoped avoidance still appears as an `avoid` proposition and uses `during`
  to list the visit segment(s) in which it is active.
- Soft proposition kinds are `prefer_near`, `prefer_far`, `prefer_relative`,
  and `prefer_path_shape`. They do not appear in the Boolean LTL formula.
- `prefer_relative` needs two `ref_ids` and may set `relation`.
- `prefer_path_shape` may include a `path_shape` object with type `through`,
  `circle`, or `follow_wall`, and optional `direction` and `fraction`.
- Use only IDs that occur verbatim in the catalog.
- Preserve every destination, ordering requirement, avoidance, spatial scope,
  route-stage scope, and soft preference stated by the instruction.
- Do not output dense cells, paths, planner parameters, Markdown, or comments.

