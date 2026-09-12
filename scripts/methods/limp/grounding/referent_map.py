"""Ground LIMP CRDs to SemPathBench map candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

from scripts.methods.limp.grounding.candidates import candidates_to_records, lookup_candidates
from scripts.methods.limp.grounding.comparators_2d import evaluate_comparator
from scripts.methods.limp.grounding.crd_parser import CRD, ParsedPredicate
from scripts.methods.limp.grounding.scene_adapter import Candidate, CandidateRegistry
from scripts.methods.limp.utils.geometry import bbox, min_cell_distance


@dataclass
class GroundingResult:
    referent: str
    status: str
    selected_candidate_id: str | None = None
    selected_candidate_ids: tuple[str, ...] = ()
    candidate_ids_before_filtering: tuple[str, ...] = ()
    candidate_ids_after_filtering: tuple[str, ...] = ()
    unsupported_comparators: tuple[str, ...] = ()
    comparator_results: list[dict[str, object]] = field(default_factory=list)
    disambiguation: dict[str, object] | None = None
    failure_reason: str | None = None

    def to_dict(self, registry: CandidateRegistry) -> dict[str, object]:
        return {
            "referent": self.referent,
            "status": self.status,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "candidate_ids_before_filtering": list(self.candidate_ids_before_filtering),
            "candidate_ids_after_filtering": list(self.candidate_ids_after_filtering),
            "candidates_after_filtering": candidates_to_records(
                registry,
                self.candidate_ids_after_filtering,
            ),
            "unsupported_comparators": list(self.unsupported_comparators),
            "comparator_results": self.comparator_results,
            "disambiguation": self.disambiguation,
            "failure_reason": self.failure_reason,
        }


def crd_to_text(crd: CRD) -> str:
    text = crd.base
    for comparator in crd.comparators:
        args = ",".join(crd_to_text(arg) for arg in comparator.args)
        text += f"::{comparator.name}({args})"
    return text


def _ground_candidate_set(
    crd: CRD,
    registry: CandidateRegistry,
    comparator_results: list[dict[str, object]],
    unsupported: set[str],
    *,
    extended: bool = True,
) -> tuple[str, ...]:
    current = set(lookup_candidates(registry, crd.base, extended=extended))
    for comparator in crd.comparators:
        arg_sets = [
            _ground_candidate_set(
                arg,
                registry,
                comparator_results,
                unsupported,
                extended=extended,
            )
            for arg in comparator.args
        ]
        filtered: set[str] = set()
        for candidate_id in current:
            candidate = registry.get(candidate_id)
            if candidate is None:
                continue
            matched_reference_ids: list[list[str]] = []
            if len(arg_sets) == 1:
                for ref_id in arg_sets[0]:
                    ref = registry.get(ref_id)
                    if ref is None:
                        continue
                    value = evaluate_comparator(candidate, comparator.name, (ref,))
                    comparator_results.append(
                        {
                            "candidate_id": candidate_id,
                            "comparator": comparator.name,
                            "reference_ids": [ref_id],
                            "spatial_frame": "global_map_frame",
                            "frame_source": "SemPathBench top-down grid convention",
                            "value": value,
                        }
                    )
                    if value is None:
                        unsupported.add(comparator.name)
                    elif value:
                        matched_reference_ids.append([ref_id])
                if matched_reference_ids:
                    filtered.add(candidate_id)
            elif len(arg_sets) == 2:
                for first_id in arg_sets[0]:
                    first = registry.get(first_id)
                    if first is None:
                        continue
                    for second_id in arg_sets[1]:
                        second = registry.get(second_id)
                        if second is None:
                            continue
                        value = evaluate_comparator(candidate, comparator.name, (first, second))
                        comparator_results.append(
                            {
                                "candidate_id": candidate_id,
                                "comparator": comparator.name,
                                "reference_ids": [first_id, second_id],
                                "spatial_frame": "global_map_frame",
                                "frame_source": "SemPathBench top-down grid convention",
                                "value": value,
                            }
                        )
                        if value is None:
                            unsupported.add(comparator.name)
                        elif value:
                            matched_reference_ids.append([first_id, second_id])
                if matched_reference_ids:
                    filtered.add(candidate_id)
            else:
                unsupported.add(comparator.name)

        # "Next to" commonly identifies the closest instance, not every
        # object inside one large fixed threshold. Preserve room-membership
        # behavior, but rank object-to-object matches by mask distance.
        if (
            extended
            and comparator.name in {"isnextto", "nextto"}
            and filtered
            and len(arg_sets) == 1
        ):
            references = [
                registry.get(reference_id)
                for reference_id in arg_sets[0]
                if registry.get(reference_id) is not None
            ]
            if references and all(reference.kind != "room" for reference in references):
                distances = {
                    candidate_id: min(
                        min_cell_distance(registry.candidates[candidate_id].cells, reference.cells)
                        for reference in references
                    )
                    for candidate_id in filtered
                }
                best_distance = min(distances.values())
                filtered = {
                    candidate_id
                    for candidate_id, candidate_distance in distances.items()
                    if candidate_distance <= best_distance + 1.0
                }
        current = filtered
    return tuple(sorted(current))


VIRTUAL_BASES = {
    "midpoint",
    "middle",
    "center",
    "centre",
    "point",
    "spot",
    "area",
    "space",
    "location",
}


def _add_virtual_candidate(
    registry: CandidateRegistry,
    *,
    category: str,
    name: str,
    cell: tuple[int, int],
    reference_ids: tuple[str, ...],
) -> str:
    digest = hashlib.sha1((name + "|" + ",".join(reference_ids)).encode("utf-8")).hexdigest()[:10]
    candidate_id = f"virtual_{digest}"
    if candidate_id not in registry.candidates:
        registry.candidates[candidate_id] = Candidate(
            candidate_id=candidate_id,
            kind="virtual",
            instance_id=-len([key for key in registry.candidates if key.startswith("virtual_")]) - 1,
            category=category,
            name=name,
            cells=(cell,),
            centroid=(float(cell[0]), float(cell[1])),
            bbox=bbox((cell,)),
            attributes=("virtual_grounding",),
        )
        registry.by_kind["virtual"] = tuple(sorted((*registry.by_kind.get("virtual", ()), candidate_id)))
        registry.by_category[category] = tuple(sorted((*registry.by_category.get(category, ()), candidate_id)))
    return candidate_id


def _virtual_between_candidate_set(
    crd: CRD,
    registry: CandidateRegistry,
    comparator_results: list[dict[str, object]],
    unsupported: set[str],
    *,
    extended: bool = True,
) -> tuple[str, ...]:
    if crd.base not in VIRTUAL_BASES:
        return ()
    if len(crd.comparators) != 1:
        return ()
    comparator = crd.comparators[0]
    if comparator.name not in {"isbetween", "between"} or len(comparator.args) != 2:
        return ()
    first_ids = _ground_candidate_set(comparator.args[0], registry, comparator_results, unsupported, extended=extended)
    second_ids = _ground_candidate_set(comparator.args[1], registry, comparator_results, unsupported, extended=extended)
    virtual_ids: list[str] = []
    for first_id in first_ids:
        first = registry.get(first_id)
        if first is None:
            continue
        for second_id in second_ids:
            second = registry.get(second_id)
            if second is None:
                continue
            row = int(round((first.centroid[0] + second.centroid[0]) / 2.0))
            col = int(round((first.centroid[1] + second.centroid[1]) / 2.0))
            virtual_id = _add_virtual_candidate(
                registry,
                category=crd.base,
                name=crd_to_text(crd),
                cell=(row, col),
                reference_ids=(first_id, second_id),
            )
            virtual_ids.append(virtual_id)
            comparator_results.append(
                {
                    "candidate_id": virtual_id,
                    "comparator": comparator.name,
                    "reference_ids": [first_id, second_id],
                    "spatial_frame": "global_map_frame",
                    "frame_source": "virtual midpoint between grounded references",
                    "value": True,
                }
            )
    return tuple(sorted(set(virtual_ids)))


def ground_crd(
    crd: CRD,
    registry: CandidateRegistry,
    *,
    extended: bool = True,
) -> GroundingResult:
    comparator_results: list[dict[str, object]] = []
    unsupported: set[str] = set()
    before = lookup_candidates(registry, crd.base, extended=extended)
    after = _ground_candidate_set(crd, registry, comparator_results, unsupported, extended=extended)
    if not before and not after:
        after = _virtual_between_candidate_set(crd, registry, comparator_results, unsupported, extended=extended)
        before = after
    text = crd_to_text(crd)
    if unsupported:
        return GroundingResult(
            referent=text,
            status="UNSUPPORTED_COMPARATOR",
            candidate_ids_before_filtering=before,
            candidate_ids_after_filtering=after,
            unsupported_comparators=tuple(sorted(unsupported)),
            comparator_results=comparator_results,
            failure_reason=f"Unsupported or unresolved comparators: {', '.join(sorted(unsupported))}",
        )
    if not after:
        return GroundingResult(
            referent=text,
            status="GROUNDING_FAILED",
            candidate_ids_before_filtering=before,
            candidate_ids_after_filtering=after,
            comparator_results=comparator_results,
            failure_reason="No candidate instance satisfies the referent descriptor.",
        )
    if len(after) > 1:
        return GroundingResult(
            referent=text,
            status="AMBIGUOUS_GROUNDING",
            candidate_ids_before_filtering=before,
            candidate_ids_after_filtering=after,
            comparator_results=comparator_results,
            failure_reason="Multiple candidates satisfy the descriptor and no explicit ranking selector is available.",
        )
    return GroundingResult(
        referent=text,
        status="GROUNDING_SUCCESS",
        selected_candidate_id=after[0],
        selected_candidate_ids=(after[0],),
        candidate_ids_before_filtering=before,
        candidate_ids_after_filtering=after,
        comparator_results=comparator_results,
    )


def ground_predicates(
    predicates: list[ParsedPredicate],
    registry: CandidateRegistry,
    *,
    extended: bool = True,
) -> tuple[dict[str, GroundingResult], list[dict[str, object]]]:
    by_referent: dict[str, GroundingResult] = {}
    records: list[dict[str, object]] = []
    for predicate in predicates:
        for referent in predicate.referents:
            key = crd_to_text(referent)
            if key not in by_referent:
                by_referent[key] = ground_crd(referent, registry, extended=extended)
            records.append(
                {
                    "encoded": predicate.encoded,
                    "predicate": predicate.predicate,
                    "referent": key,
                    "grounding_status": by_referent[key].status,
                    "selected_candidate_id": by_referent[key].selected_candidate_id,
                }
            )
    return by_referent, records
