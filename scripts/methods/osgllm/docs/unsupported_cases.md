# Unsupported Cases

This first local adapter preserves the Boolean LTL planning boundary of OSG-LLM.
It logs, but does not optimize, soft/path-shape constraints.

| case | example | status | reason | possible future extension |
| --- | --- | --- | --- | --- |
| `near_preference` | try to walk closer to the sofa | logged unsupported | soft preference, not a hard Boolean LTL goal | add weighted cost shaping or MHA*/AMRA* inadmissible queue |
| `far_preference` | stay farther away from the sink | logged unsupported | soft preference, not a hard Boolean LTL goal | add distance-field cost term |
| `relative_preference` | closer to A than B | logged unsupported when soft | requires metric preference over a segment | add temporal cost monitor |
| `path_shape_preference` | walk around the room, circle, along the wall | logged unsupported | path-shape constraints are not native OSG-LTL propositions | add virtual regions and path-shape monitors |
| `door_preference` | enter from the door farther from the sofa | logged unsupported | doors are not robustly represented as first-class regions yet | derive doorway regions from room adjacency boundaries |

