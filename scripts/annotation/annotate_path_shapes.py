#!/usr/bin/env python3
"""Local browser UI for canonical path-shape trajectory annotation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import webbrowser
import warnings
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation.path_shape_schema import (
    PATH_SHAPE_ANNOTATION_FIELD,
    has_path_shape_annotation,
    normalize_path_shape_annotation,
)
from scripts.evaluation.evaluate_prediction import load_instruction_by_id
from scripts.evaluation.evaluation_metrics import build_instruction_metric_cache
from scripts.make_instruction.make_instruction import (
    REPO_ROOT,
    hard_constraint_hit_indexes,
    instruction_files_directory,
    list_map_summaries,
    load_map_state,
    scoped_trajectory,
    trajectory_inside_constraint_region,
    validate_route,
)
from scripts.methods.util.instructions import (
    DEFAULT_INSTRUCTION_SET,
    INSTRUCTION_SET_CHOICES,
    instruction_id_from_payload,
    map_id_matches_instruction_set,
    normalize_instruction_set,
)

def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def path_text(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def json_for_html(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def add_human_path_shape_references(
    instruction: dict[str, object],
) -> dict[str, object]:
    """Add runtime-only scoped human references for annotation QA."""
    trajectory = instruction.get("human_expert_trajectory")
    hard_constraints = instruction.get("hard_constraints")
    soft_constraints = instruction.get("soft_constraints")
    if (
        not isinstance(trajectory, list)
        or not isinstance(hard_constraints, list)
        or not isinstance(soft_constraints, list)
    ):
        return instruction

    sample = {
        "expert_route": trajectory,
        "hard_constraints": hard_constraints,
        "soft_constraints": soft_constraints,
    }
    hit_indexes = hard_constraint_hit_indexes(sample)
    for constraint in soft_constraints:
        if (
            not isinstance(constraint, dict)
            or constraint.get("preference_type") != "path_shape_preference"
        ):
            continue
        active = scoped_trajectory(sample, constraint, hit_indexes)
        constraint["_human_reference_trajectory"] = (
            trajectory_inside_constraint_region(active, constraint)
        )
    return instruction


def shape_constraints(instruction: Mapping[str, object]) -> list[dict[str, object]]:
    raw = instruction.get("soft_constraints")
    if not isinstance(raw, list):
        return []
    result: list[dict[str, object]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping) or item.get("preference_type") != "path_shape_preference":
            continue
        identifier = item.get("constraint_id")
        if not isinstance(identifier, str) or not identifier.strip():
            identifier = f"path_shape_{index}"
        result.append(
            {
                "constraint_id": identifier,
                "label": item.get("label") or "Path Shape",
                "shape": item.get("shape") or "unknown",
                "center": item.get("center"),
                "radius": item.get("radius"),
                "width": item.get("width"),
                "height": item.get("height"),
                "cells": item.get("cells") if isinstance(item.get("cells"), list) else [],
                "scope": item.get("scope"),
                "reference_trajectory": item.get("reference_trajectory") if isinstance(item.get("reference_trajectory"), list) else [],
            }
        )
    return result


def instruction_records(map_id: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(instruction_files_directory(map_id).glob("instruction_*.json")):
        try:
            instruction = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(instruction, dict):
            continue
        constraints = shape_constraints(instruction)
        if not constraints:
            continue
        instruction_id = instruction_id_from_payload(path, instruction)
        complete = sum(
            has_path_shape_annotation(item)
            for item in instruction.get("soft_constraints", [])
            if isinstance(item, Mapping)
            and item.get("preference_type") == "path_shape_preference"
        )
        records.append(
            {
                "instruction_id": instruction_id,
                "instruction": instruction.get("instruction") or "",
                "difficulty_level": instruction.get("difficulty_level") or "unknown",
                "shape_count": len(constraints),
                "annotated_count": complete,
            }
        )
    return records


def maps_with_shapes(instruction_set: str) -> list[dict[str, object]]:
    selected = normalize_instruction_set(instruction_set)
    output: list[dict[str, object]] = []
    for summary in list_map_summaries():
        map_id = str(summary["map_id"])
        if not map_id_matches_instruction_set(map_id, selected):
            continue
        instructions = instruction_records(map_id)
        if instructions:
            output.append(
                {
                    "map_id": map_id,
                    "shape_instruction_count": len(instructions),
                    "shape_constraint_count": sum(
                        int(item["shape_count"]) for item in instructions
                    ),
                    "annotated_count": sum(
                        int(item["annotated_count"]) for item in instructions
                    ),
                }
            )
    return output


def client_state(
    requested_map: str | None,
    instruction_set: str,
) -> dict[str, object]:
    maps = maps_with_shapes(instruction_set)
    if requested_map:
        map_id = str(load_map_state(requested_map)["map_key"])
    elif maps:
        map_id = str(maps[0]["map_id"])
    else:
        raise ValueError("No instructions with path_shape_preference found for this split.")
    map_state = load_map_state(map_id)
    map_id = str(map_state["map_key"])
    return {
        "instruction_set": normalize_instruction_set(instruction_set),
        "current_map_id": map_id,
        "map_state": map_state,
        "maps": maps,
        "instructions": instruction_records(map_id),
        "storage": "embedded in instruction soft_constraints",
    }


def find_constraint(instruction: Mapping[str, object], constraint_id: str) -> dict[str, object]:
    for constraint in instruction.get("soft_constraints", []):
        if (
            isinstance(constraint, dict)
            and constraint.get("preference_type") == "path_shape_preference"
            and constraint.get("constraint_id") == constraint_id
        ):
            return constraint
    raise ValueError(f"Unknown path-shape constraint: {constraint_id}")


def points(value: object, label: str, grid_size: int, minimum: int) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of [row, col] points.")
    output: list[list[int]] = []
    for index, point in enumerate(value):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            raise ValueError(f"{label}[{index}] must be [row, col].")
        row, col = point
        if not isinstance(row, int) or not isinstance(col, int):
            raise ValueError(f"{label}[{index}] must contain integers.")
        if not (0 <= row < grid_size and 0 <= col < grid_size):
            raise ValueError(f"{label}[{index}] is outside the map.")
        if not output or output[-1] != [row, col]:
            output.append([row, col])
    if len(output) < minimum:
        raise ValueError(f"{label} needs at least {minimum} points.")
    return output


def write_instruction(path: Path, instruction: Mapping[str, object]) -> None:
    """Atomically replace one instruction JSON file."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(instruction, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_embedded_annotation(
    map_id: str,
    instruction_id: str,
    constraint_id: str,
) -> dict[str, object] | None:
    instruction, _ = load_instruction_by_id(map_id, instruction_id)
    constraint = find_constraint(instruction, constraint_id)
    annotation = constraint.get(PATH_SHAPE_ANNOTATION_FIELD)
    return dict(annotation) if isinstance(annotation, Mapping) else None


def save_embedded_annotation(payload: Mapping[str, object]) -> dict[str, object]:
    map_id = payload.get("map_id")
    instruction_id = payload.get("instruction_id")
    constraint_id = payload.get("constraint_id")
    if not all(
        isinstance(item, str) and item.strip()
        for item in (map_id, instruction_id, constraint_id)
    ):
        raise ValueError("map_id, instruction_id, and constraint_id are required.")

    map_state = load_map_state(str(map_id))
    map_id = str(map_state["map_key"])
    instruction, instruction_path = load_instruction_by_id(
        map_id,
        str(instruction_id),
    )
    constraint = find_constraint(instruction, str(constraint_id))
    size = int(map_state["grid_size"])
    trajectory = validate_route(
        points(
            payload.get("shape_reference_trajectory"),
            "shape_reference_trajectory",
            size,
            2,
        ),
        size,
        label="shape_reference_trajectory",
    )
    controls = points(
        payload.get("ordered_control_points", []),
        "ordered_control_points",
        size,
        0,
    )
    shape_type = str(payload.get("shape_type") or "expert_polyline").strip()
    direction = str(payload.get("direction") or "unspecified").strip()
    annotator = str(payload.get("annotator_id") or "").strip()
    notes = str(payload.get("notes") or "").strip()
    timestamp = now()

    raw_previous = constraint.get(PATH_SHAPE_ANNOTATION_FIELD)
    previous = (
        normalize_path_shape_annotation(raw_previous, grid_size=size)
        if raw_previous is not None
        else None
    )
    history: list[object] = []
    if previous is not None:
        previous_history = previous.get("history", [])
        if isinstance(previous_history, list):
            history.extend(previous_history)
        history.append(
            {
                key: value
                for key, value in previous.items()
                if key != "history"
            }
        )

    record = normalize_path_shape_annotation(
        {
            "version": 1,
            "annotation_type": "canonical_path_shape_reference",
            "shape_spec": {
                "type": shape_type or "expert_polyline",
                "direction": direction or "unspecified",
                "ordered_control_points": controls,
            },
            "shape_reference_trajectory": trajectory,
            "shape_reference_source": "human_expert_canonical_v1",
            "annotator_id": annotator,
            "notes": notes,
            "created_at": (
                previous.get("created_at", timestamp)
                if previous is not None
                else timestamp
            ),
            "updated_at": timestamp,
            "history": history,
        },
        grid_size=size,
    )
    constraint[PATH_SHAPE_ANNOTATION_FIELD] = record
    # Keep the established metric field synchronized for older consumers.
    constraint["reference_trajectory"] = trajectory
    constraint["reference_waypoint_count"] = len(trajectory)
    constraint["reference_metric_status"] = "computed"
    instruction["updated_at"] = timestamp
    write_instruction(instruction_path, instruction)
    try:
        build_instruction_metric_cache(instruction_path, instruction, map_state)
    except Exception as exc:
        warnings.warn(
            f"Saved the annotation, but could not refresh its metric cache: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "message": "Saved canonical path-shape reference in the instruction.",
        "annotation": record,
        "path": path_text(instruction_path),
        "newly_annotated": previous is None,
    }


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SemPathBench Shape Annotation</title><style>
:root{--bg:#f4f7f5;--panel:#fff;--line:#d3dbd5;--ink:#1b2920;--muted:#66736b;--green:#0a6d54;--orange:#ad4b29}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px Arial,sans-serif}.app{display:grid;grid-template-columns:280px minmax(0,1fr) 335px;gap:12px;min-height:100vh;padding:12px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px;min-width:0}.scroll{max-height:calc(100vh - 24px);overflow:auto}.main{display:flex;flex-direction:column;gap:9px}.canvas-wrap{max-height:calc(100vh - 150px);overflow:auto;border:1px solid var(--line);border-radius:7px;background:#e6ebe7}canvas{display:block;image-rendering:pixelated;cursor:crosshair}h1{font-size:20px;margin:0 0 6px}h2{font-size:15px;margin:13px 0 6px}.help{color:var(--muted);font-size:12px;line-height:1.4}.item{border:1px solid var(--line);padding:8px;border-radius:7px;margin:6px 0;cursor:pointer}.item.active{border-color:var(--green);background:#e8f5ef}.badge{float:right;background:#e8eee9;padding:2px 6px;border-radius:10px;font-size:11px}label{display:block;font-size:12px;font-weight:bold;margin:9px 0 4px}input,select,textarea,button{width:100%;font:inherit}input,select,textarea{border:1px solid var(--line);border-radius:6px;padding:7px}textarea{min-height:68px;resize:vertical}button{border:0;border-radius:6px;padding:8px;background:var(--green);color:#fff;cursor:pointer;margin-top:6px}button.secondary{background:#e7eeea;color:#1a4b3b}button.warn{background:var(--orange)}button.active{outline:3px solid #a8ddc8}.row{display:flex;gap:7px}.row>*{flex:1}.status{padding:8px;background:#edf5f0;border-radius:6px;min-height:38px}.legend{font-size:12px;display:flex;flex-wrap:wrap;gap:9px}.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:3px}.check{display:flex;align-items:center;gap:6px;font-weight:normal}.check input{width:auto}@media(max-width:1000px){.app{grid-template-columns:250px 1fr}.right{grid-column:1/-1}.scroll{max-height:none}}@media(max-width:700px){.app{display:block}.panel{margin-bottom:10px}}</style></head><body><main class="app">
<aside class="panel scroll"><h1>Canonical Shape</h1><p class="help">Create an independent expert reference; the human demonstration is hidden unless used for QA.</p><label>Map</label><select id="maps"></select><h2>Instructions</h2><div id="items"></div></aside>
<section class="panel main"><div class="row"><button id="draw" class="active">Draw trajectory</button><button id="control" class="secondary">Control point</button><button id="undo" class="secondary">Undo</button><button id="reset" class="warn">Reset</button></div><div class="legend"><span><i class="dot" style="background:#edc547"></i>shape region</span><span><i class="dot" style="background:#0a6d54"></i>canonical route</span><span><i class="dot" style="background:#8e3ad0"></i>ordered controls</span><span><i class="dot" style="background:#d75045"></i>human demo</span></div><div class="canvas-wrap"><canvas id="canvas"></canvas></div><div id="inspect" class="status">Furniture: click a colored object on the map to identify it.</div><div id="status" class="status">Select an instruction.</div><div id="score" class="status">nDTW QA: select a path-shape constraint.</div></section>
<aside class="panel scroll right"><h2 id="heading">Constraint</h2><p id="instruction" class="help"></p><label>Path-shape constraint</label><select id="constraints"></select><p id="info" class="help"></p><label class="check"><input id="human" type="checkbox">Show human demonstration for QA</label><label class="check"><input id="rooms" type="checkbox" checked>Show room map</label><label class="check"><input id="objects" type="checkbox" checked>Show object map</label><label>Annotator ID</label><input id="annotator" placeholder="expert_01"><label>Canonical type</label><input id="kind" value="expert_polyline"><label>Direction</label><select id="direction"><option>unspecified</option><option>clockwise</option><option>counterclockwise</option><option>forward</option><option>reverse</option></select><label>Notes</label><textarea id="notes"></textarea><label>Zoom <span id="zoomText">1.5x</span></label><input id="zoom" type="range" min="1" max="5" value="1.5" step=".5"><button id="save">Validate & save</button><p class="help">Draw mode: click/drag through free cells. Control mode: click semantic milestones in order. Right-click removes the last item in the active mode.</p></aside>
</main><script>
let state=__STATE__,map=state.map_state,items=state.instructions||[],active=items[0]?.instruction_id||null,data=null,route=[],controls=[],mode="draw",painting=false,last=null,zoom=1.5,showHuman=false,showRooms=true,showObjects=true;const $=x=>document.getElementById(x),canvas=$("canvas"),ctx=canvas.getContext("2d");
function api(url,body){return fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(async r=>{let t=await r.text();if(!r.ok)throw Error(t);return JSON.parse(t)})}function msg(t){$("status").textContent=t}function k(p){return p[0]+","+p[1]}function current(){return items.find(x=>x.instruction_id===active)}function cs(){return(data?.soft_constraints||[]).filter(x=>x.preference_type==="path_shape_preference")}function cid(){return $("constraints").value}function constraint(){return cs().find(x=>x.constraint_id===cid())}function free(r,c){return map.layers.occupancy?.[r]?.[c]===0||map.layers.occupancy?.[r]?.[c]==="free"}
function objectAt(p){let id=map.layers.object_instance?.[p[0]]?.[p[1]];return(map.object_instances||[]).find(x=>Number(x.id)===Number(id))||null}
function inspectObjectAt(p){let object=objectAt(p);if(!object)return false;let landmark=(data?.objects||[]).find(x=>Number(x.object_id)===Number(object.id)),legend=map.layer_legends?.object_categories?.[object.category]||{},name=landmark?.name||object.name||object.label||legend.label||String(object.category||"Unknown object"),category=legend.label||String(object.category||"unknown").replaceAll("_"," "),roomId=map.layers.room?.[p[0]]?.[p[1]],room=(map.room_instances||[]).find(x=>Number(x.id)===Number(roomId)),roomLegend=map.layer_legends?.room_categories?.[room?.category]||{},roomName=room?.name||roomLegend.label||String(room?.category||"unknown").replaceAll("_"," ");document.getElementById("inspect").textContent="Furniture: "+name+" · category="+category+" · object id="+object.id+" · room="+roomName+" · cell=["+p[0]+", "+p[1]+"]";return true}
function clearInspection(){document.getElementById("inspect").textContent="Furniture: click a colored object on the map to identify it."}
function cells(a,b){let n=Math.max(Math.abs(a[0]-b[0]),Math.abs(a[1]-b[1])),o=[];for(let i=0;i<=n;i++){let t=n?i/n:0,p=[Math.round(a[0]+(b[0]-a[0])*t),Math.round(a[1]+(b[1]-a[1])*t)];if(!o.length||k(o[o.length-1])!==k(p))o.push(p)}return o}function eventCell(e){let b=canvas.getBoundingClientRect(),r=Math.floor((e.clientY-b.top)/zoom),c=Math.floor((e.clientX-b.left)/zoom);return r>=0&&c>=0&&r<map.grid_size&&c<map.grid_size?[r,c]:null}
function add(p){if(!free(...p)){msg("Blocked cell.");return}let a=route.length?cells(route[route.length-1],p):[p];if(a.some(q=>!free(...q))){msg("Stroke crosses an obstacle; add intermediate points around it.");return}for(let q of a)if(!route.length||k(route[route.length-1])!==k(q))route.push(q);draw()}
function inShape(p,c){if(!c)return true;if(c.shape==="freeform"){if(!c._shapeCells)c._shapeCells=new Set((c.cells||[]).map(x=>x[0]+","+x[1]));return c._shapeCells.has(k(p))}if(!Array.isArray(c.center))return true;if(c.shape==="circle")return Math.hypot(p[0]-c.center[0],p[1]-c.center[1])<=Number(c.radius||0);if(c.shape==="rectangle")return Math.abs(p[0]-c.center[0])<=Number(c.height||0)/2&&Math.abs(p[1]-c.center[1])<=Number(c.width||0)/2;return true}
function ndtw(pred,ref){if(!pred?.length||!ref?.length)return null;let prev=Array(ref.length+1).fill(Infinity);prev[0]=0;for(let i=1;i<=pred.length;i++){let cur=Array(ref.length+1).fill(Infinity);for(let j=1;j<=ref.length;j++){let d=Math.hypot(pred[i-1][0]-ref[j-1][0],pred[i-1][1]-ref[j-1][1]);cur[j]=Math.min(prev[j],cur[j-1],prev[j-1])+Math.max(0,d-10)}prev=cur}return Math.exp(-prev[ref.length]/(4*Math.max(ref.length,1)))}
function updateScore(){let c=constraint(),h=c?._human_reference_trajectory||[],candidate=route.filter(p=>inShape(p,c)),box=$("score");if(candidate.length<2||h.length<2){box.textContent="nDTW QA: draw at least two canonical points in the shape region.";return}let humanToCanonical=ndtw(h,candidate),canonicalToHuman=ndtw(candidate,h);box.innerHTML="Human demo → canonical (QA): <strong>"+humanToCanonical.toFixed(3)+"</strong><br>Canonical → human (current evaluator orientation): "+canonicalToHuman.toFixed(3)+"<br><span class='help'>shape-region points: canonical="+candidate.length+", human="+h.length+".</span>"}
function refresh(){draw();updateScore()}
function line(points,color,w){if(!points?.length)return;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=w;ctx.lineJoin="round";ctx.lineCap="round";ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo((p[1]+.5)*zoom,(p[0]+.5)*zoom):ctx.moveTo((p[1]+.5)*zoom,(p[0]+.5)*zoom));ctx.stroke();ctx.restore()}
function marker(p,label,color){if(!p)return;ctx.save();ctx.fillStyle="#fff";ctx.strokeStyle=color;ctx.lineWidth=Math.max(1,zoom*.7);ctx.beginPath();ctx.arc((p[1]+.5)*zoom,(p[0]+.5)*zoom,Math.max(4,zoom*3),0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle=color;ctx.font=Math.max(10,zoom*5)+"px Arial";ctx.textAlign="left";ctx.textBaseline="bottom";ctx.fillText(label,(p[1]+.5)*zoom+4,(p[0]+.5)*zoom-3);ctx.restore()}
function drawRegion(c){if(!c)return;ctx.save();ctx.fillStyle="rgba(237,197,71,.20)";ctx.strokeStyle="#aa7900";ctx.lineWidth=Math.max(1,zoom*.7);if(c.shape==="freeform"){for(let p of(c.cells||[]))ctx.fillRect(p[1]*zoom,p[0]*zoom,zoom,zoom)}else if(c.shape==="circle"&&Array.isArray(c.center)){ctx.beginPath();ctx.arc((c.center[1]+.5)*zoom,(c.center[0]+.5)*zoom,(Number(c.radius)||0)*zoom,0,Math.PI*2);ctx.fill();ctx.stroke()}else if(c.shape==="rectangle"&&Array.isArray(c.center)){let w=Number(c.width)||0,h=Number(c.height)||0,x=(c.center[1]-w/2)*zoom,y=(c.center[0]-h/2)*zoom;ctx.fillRect(x,y,w*zoom,h*zoom);ctx.strokeRect(x,y,w*zoom,h*zoom)}ctx.restore()}
function draw(){let n=map.grid_size,rooms=map.layers.room,objects=map.layers.object_instance,roomById=new Map((map.room_instances||[]).map(x=>[x.id,x])),objectById=new Map((map.object_instances||[]).map(x=>[x.id,x]));canvas.width=n*zoom;canvas.height=n*zoom;ctx.fillStyle="#fff";ctx.fillRect(0,0,canvas.width,canvas.height);for(let r=0;r<n;r++)for(let c=0;c<n;c++){if(!free(r,c)){ctx.fillStyle="#202522";ctx.fillRect(c*zoom,r*zoom,zoom,zoom)}else{let room=roomById.get(rooms?.[r]?.[c]),col=map.layer_legends?.room_categories?.[room?.category]?.color;if(showRooms&&col){ctx.fillStyle=col+"55";ctx.fillRect(c*zoom,r*zoom,zoom,zoom)}}let object=objectById.get(objects?.[r]?.[c]),objColor=map.layer_legends?.object_categories?.[object?.category]?.color;if(showObjects&&object&&objColor){ctx.fillStyle=objColor+"d0";ctx.fillRect(c*zoom,r*zoom,zoom,zoom)}}drawRegion(constraint());if(showHuman){let h=constraint()?._human_reference_trajectory||[];line(h,"#d75045",Math.max(1,zoom*.75));if(h.length){marker(h[0],"H-start","#d75045");marker(h[h.length-1],"H-end","#8f211c")}}line(route,"#0a6d54",Math.max(2,zoom*.9));controls.forEach((p,i)=>{ctx.fillStyle="#8e3ad0";ctx.beginPath();ctx.arc((p[1]+.5)*zoom,(p[0]+.5)*zoom,Math.max(3,zoom*2.3),0,Math.PI*2);ctx.fill();ctx.fillStyle="#fff";ctx.font=Math.max(9,zoom*5)+"px Arial";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(String(i+1),(p[1]+.5)*zoom,(p[0]+.5)*zoom)});let s=data?.start_pose;if(s){ctx.fillStyle="#2468bd";ctx.fillRect(s.col*zoom,s.row*zoom,zoom,zoom)}}
function renderItems(){let root=$("items");root.innerHTML="";for(let x of items){let d=document.createElement("div");d.className="item "+(x.instruction_id===active?"active":"");d.innerHTML="<span class='badge'>"+x.annotated_count+"/"+x.shape_count+"</span><strong>"+x.instruction_id+"</strong><br><span class='help'>"+x.instruction+"</span>";d.onclick=()=>select(x.instruction_id);root.appendChild(d)}}
function renderConstraints(){let s=$("constraints");s.innerHTML="";for(let c of cs()){let o=document.createElement("option");o.value=c.constraint_id;o.textContent=(c.label||"Path Shape")+" · "+c.shape+" · "+c.constraint_id;s.appendChild(o)}s.onchange=load;info()}
function info(){let c=constraint();$("heading").textContent=c?.label||"Constraint";$("info").textContent=c?"shape="+c.shape+" · scope="+JSON.stringify(c.scope||{})+" · human points="+(c._human_reference_trajectory||[]).length:""}
async function select(id){active=id;renderItems();clearInspection();try{let r=await api("/instruction",{map_id:state.current_map_id,instruction_id:id});data=r.instruction;$("instruction").textContent=data.instruction||"";renderConstraints();await load()}catch(e){msg(e.message)}}
async function load(){route=[];controls=[];info();let c=constraint();if(!c){refresh();return}try{let r=await api("/annotation",{map_id:state.current_map_id,instruction_id:active,constraint_id:c.constraint_id});let a=r.annotation;if(a){route=a.shape_reference_trajectory||[];controls=a.shape_spec?.ordered_control_points||[];$("annotator").value=a.annotator_id||"";$("kind").value=a.shape_spec?.type||"expert_polyline";$("direction").value=a.shape_spec?.direction||"unspecified";$("notes").value=a.notes||"";msg("Loaded existing reference. Saving keeps a history backup.")}else msg("No reference yet. Draw independently.")}catch(e){msg(e.message)}refresh()}
function modes(m){mode=m;$("draw").className=m==="draw"?"active":"secondary";$("control").className=m==="control"?"active":"secondary";msg(m==="draw"?"Draw a dense valid route.":"Click ordered control points.")}
function renderMaps(){let s=$("maps");s.innerHTML="";for(let x of state.maps){let o=document.createElement("option");o.value=x.map_id;o.textContent=x.map_id+" ("+x.annotated_count+"/"+x.shape_constraint_count+")";o.selected=x.map_id===state.current_map_id;s.appendChild(o)}s.onchange=async()=>{try{let r=await api("/open",{map_id:s.value});state=r.state;map=state.map_state;items=state.instructions||[];active=items[0]?.instruction_id||null;data=null;route=[];controls=[];clearInspection();renderMaps();renderItems();if(active)await select(active);else refresh()}catch(e){msg(e.message)}}}
canvas.onmousedown=e=>{if(e.button!==0)return;let p=eventCell(e);if(!p)return;if(inspectObjectAt(p)){painting=false;return}if(mode==="control"){if(free(...p)){controls.push(p);draw();msg("Added control point "+controls.length)}return}painting=true;last=p;add(p)};canvas.onmousemove=e=>{if(!painting||mode!=="draw")return;let p=eventCell(e);if(p&&k(p)!==k(last)){last=p;add(p)}};window.onmouseup=()=>{painting=false;updateScore()};canvas.oncontextmenu=e=>{e.preventDefault();if(mode==="control")controls.pop();else route.pop();refresh()};
$("draw").onclick=()=>modes("draw");$("control").onclick=()=>modes("control");$("undo").onclick=()=>{if(mode==="control")controls.pop();else route.pop();refresh()};$("reset").onclick=()=>{route=[];controls=[];refresh();msg("Cleared unsaved drawing.")};$("human").onchange=e=>{showHuman=e.target.checked;refresh()};$("rooms").onchange=e=>{showRooms=e.target.checked;draw()};$("objects").onchange=e=>{showObjects=e.target.checked;draw()};$("zoom").oninput=e=>{zoom=Number(e.target.value);$("zoomText").textContent=zoom+"x";draw()};$("save").onclick=async()=>{let c=constraint();if(!c)return;try{let r=await api("/save",{map_id:state.current_map_id,instruction_id:active,constraint_id:c.constraint_id,shape_reference_trajectory:route,ordered_control_points:controls,annotator_id:$("annotator").value,shape_type:$("kind").value,direction:$("direction").value,notes:$("notes").value});msg(r.message+" "+r.path);let x=current();if(x&&r.newly_annotated)x.annotated_count=Math.min(x.shape_count,(x.annotated_count||0)+1);if(c){c.path_shape_annotation=r.annotation;c.reference_trajectory=r.annotation.shape_reference_trajectory||[];}renderItems()}catch(e){msg("Not saved: "+e.message)}};
renderMaps();renderItems();if(active)select(active);else draw();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    state: dict[str, object] = {}

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self) -> dict[str, object]:
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        value = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object.")
        return value

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = HTML.replace("__STATE__", json_for_html(self.state)).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self.body()
            if self.path == "/open":
                map_id = payload.get("map_id")
                if not isinstance(map_id, str):
                    raise ValueError("map_id is required.")
                self.state = client_state(map_id, str(self.state["instruction_set"]))
                self.send_json({"state": self.state})
            elif self.path == "/instruction":
                map_id, instruction_id = payload.get("map_id"), payload.get("instruction_id")
                if not isinstance(map_id, str) or not isinstance(instruction_id, str):
                    raise ValueError("map_id and instruction_id are required.")
                instruction, _ = load_instruction_by_id(map_id, instruction_id)
                self.send_json({"instruction": add_human_path_shape_references(instruction)})
            elif self.path == "/annotation":
                map_id, instruction_id, constraint_id = payload.get("map_id"), payload.get("instruction_id"), payload.get("constraint_id")
                if not all(isinstance(item, str) for item in (map_id, instruction_id, constraint_id)):
                    raise ValueError("map_id, instruction_id, and constraint_id are required.")
                self.send_json({"annotation": load_embedded_annotation(map_id, instruction_id, constraint_id)})
            elif self.path == "/save":
                self.send_json(save_embedded_annotation(payload))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            body = str(exc).encode("utf-8")
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate canonical path-shape trajectories in a local browser UI.")
    parser.add_argument("--map-id", default=None)
    parser.add_argument("--set", dest="instruction_set", choices=INSTRUCTION_SET_CHOICES, default=DEFAULT_INSTRUCTION_SET)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Handler.state = client_state(args.map_id, args.instruction_set)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving path-shape annotation UI at {url}")
    print("Saving path-shape annotations inside instruction JSON files")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

