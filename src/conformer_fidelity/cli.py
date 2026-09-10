"""Conformer generation, selection, evaluation, source auditing and analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__


def parser():
    app = argparse.ArgumentParser(prog="conformer-fidelity", description=__doc__)
    app.add_argument("--version", action="version", version=__version__)
    commands = app.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate from a specified 2D molecule")
    generate.add_argument("--method", choices=("etkdg", "conforge", "etflow", "avgflow"), default="etkdg")
    generate.add_argument("--smiles", required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--count", type=int, default=20)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--source", type=Path)
    generate.add_argument("--checkpoint", type=Path)
    generate.add_argument("--config")
    generate.add_argument("--device", default="cpu")
    generate.add_argument("--timeout-ms", type=int, default=60000)

    relax = commands.add_parser("relax", help="MMFF94s postprocessing with fixed output slots")
    relax.add_argument("--input", type=Path, required=True)
    relax.add_argument("--output", type=Path, required=True)
    relax.add_argument("--iterations", type=int, default=200)

    select = commands.add_parser("select", help="Target-blind selection from a fixed candidate pool")
    select.add_argument("--input", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--method", choices=("facility", "energy_facility", "energy", "kcenter",
                                             "energy_rmsd05", "energy_rmsd10"), default="facility")
    select.add_argument("--count", type=int, default=20, help="Output slots, up to 20; missing slots stay missing")

    score = commands.add_parser("score", help="Score finite/original/qualified recovery against a reference")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--reference", type=Path, required=True)
    score.add_argument("--reference-index", type=int, default=0)
    score.add_argument("--requested", type=int, default=20, help="Requested slot denominator, including failures")
    score.add_argument("--output", type=Path, required=True)

    identity = commands.add_parser("identity", help="Measure specified stereo independently on returned slots")
    identity.add_argument("--input", type=Path, required=True)
    identity.add_argument("--output", type=Path, required=True)

    audit = commands.add_parser("audit-conditioning", help="Measure upstream feature tensors on specified stereo pairs")
    audit.add_argument("--method", choices=("etflow", "avgflow"), required=True)
    audit.add_argument("--source", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--seed", type=int, default=0)

    statistics = commands.add_parser("statistics", help="Analyze already aggregated per-scaffold metrics")
    statistics.add_argument("--input", type=Path, required=True)
    statistics.add_argument("--output", type=Path, required=True, help="New output directory")
    statistics.add_argument("--spec", type=Path, help="Explicit contrasts, domains and stratum weights in JSON")
    statistics.add_argument("--draws", type=int, default=50000)
    statistics.add_argument("--seed", type=int, default=0)

    export = commands.add_parser("export", help="Export per-scaffold metrics and an optional statistics result")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--statistics", type=Path)

    demo = commands.add_parser("demo", help="CPU workflow using a generated reference conformer")
    demo.add_argument("--output", type=Path, required=True)
    return app


def _generate(args):
    from .artifacts import save_ensemble

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.method == "etkdg":
        from .generators.etkdg import generate
        result = generate(args.smiles, count=args.count, seed=args.seed)
    elif args.method == "conforge":
        from .generators.conforge import generate
        result = generate(args.smiles, count=args.count, seed=args.seed,
                          timeout_ms=args.timeout_ms, source=args.source)
    else:
        if args.source is None or args.checkpoint is None:
            raise ValueError("Neural backends require local --source and --checkpoint paths")
        options = dict(source=args.source, checkpoint=args.checkpoint, device=args.device)
        if args.method == "etflow":
            from .generators.etflow import ETFlowGenerator
            if args.config is not None:
                options["config"] = args.config
            generator = ETFlowGenerator(**options)
        else:
            from .generators.avgflow import AvgFlowGenerator
            if args.config is None:
                raise ValueError("AvgFlow also requires an explicit --config YAML path")
            generator = AvgFlowGenerator(**options, config=Path(args.config))
        result = generator.generate(args.smiles, count=args.count, seed=args.seed)
    save_ensemble(args.output, args.smiles, result)
    return args.output


def _score(args):
    from .artifacts import load_ensemble, save_json
    from .evaluation import score_pool

    candidate, reference = load_ensemble(args.input), load_ensemble(args.reference)
    # SMILES order defines the index mapping. Equivalent canonical graphs alone
    # do not establish that two coordinate arrays have the same atom order.
    if candidate["smiles"] != reference["smiles"]:
        raise ValueError("Reference and candidate must use the same atom-ordered input SMILES")
    index = args.reference_index
    if index < 0 or index >= len(reference["coordinates"]) or not reference["present_mask"][index]:
        raise ValueError("Reference index must identify a present reference slot")
    result = score_pool(candidate["smiles"], candidate["coordinates"],
                        reference["coordinates"][index], present_mask=candidate["present_mask"],
                        denominator=args.requested)
    save_json(args.output, result)
    return args.output


def _select(args):
    import numpy as np
    from .artifacts import load_ensemble, save_ensemble
    from .selection import select_conformers

    if args.output.exists():
        raise FileExistsError(args.output)
    candidate = load_ensemble(args.input)
    selection = select_conformers(candidate["smiles"], candidate["coordinates"],
                                 method=args.method, maximum=args.count,
                                 present_mask=candidate["present_mask"])
    indices = selection["indices"]
    coordinates = np.full((args.count,) + candidate["coordinates"].shape[1:], np.nan)
    coordinates[:len(indices)] = candidate["coordinates"][indices]
    present = np.arange(args.count) < len(indices)
    save_ensemble(args.output, candidate["smiles"],
                  dict(coordinates=coordinates, present_mask=present, selection=selection,
                       operation="selection", source_artifact=str(args.input),
                       requested=args.count, output_count=len(indices), target_access=False,
                       refill=False, coordinates_changed=False))
    return args.output


def _identity(args):
    import numpy as np
    from .artifacts import load_ensemble, save_json
    from .chemistry import independent_stereo

    candidate = load_ensemble(args.input)
    rows = []
    for index, (xyz, present) in enumerate(zip(candidate["coordinates"], candidate["present_mask"])):
        finite = bool(np.isfinite(xyz).all())
        if present and finite:
            row = independent_stereo(candidate["smiles"], xyz)
        else:
            row = dict(stereo_evaluated=False, specified_stereo_retained=None,
                       stereo_error="missing_output" if not present else "nonfinite_coordinates")
        rows.append(dict(slot=index, present=bool(present), finite=finite, **row))
    save_json(args.output, dict(smiles=candidate["smiles"], requested_slots=len(rows), slots=rows))
    return args.output


def _demo(output):
    from .artifacts import save_ensemble, save_json
    from .generators.etkdg import generate, relax

    if output.exists():
        raise FileExistsError("Choose a new demo output directory")
    output.mkdir(parents=True)
    smiles = "CCO"
    candidates = generate(smiles, count=20, seed=7)
    save_ensemble(output / "candidates.npz", smiles, candidates)
    save_ensemble(output / "postprocessed.npz", smiles,
                  relax(smiles, candidates["coordinates"], present_mask=candidates["present_mask"], max_iterations=200))
    _score(argparse.Namespace(input=output / "candidates.npz", reference=output / "candidates.npz",
                             reference_index=0, requested=20, output=output / "scores.json"))
    _identity(argparse.Namespace(input=output / "candidates.npz", output=output / "identity.json"))
    # The first generated conformer serves as the workflow example's reference.
    save_json(output / "example.json", dict(synthetic=True, benchmark=False,
              reference="first generated candidate", smiles=smiles))
    return output


def main(argv=None):
    app = parser()
    args = app.parse_args(argv)
    try:
        if args.command == "generate":
            output = _generate(args)
        elif args.command == "relax":
            from .artifacts import load_ensemble, save_ensemble
            from .generators.etkdg import relax
            candidate = load_ensemble(args.input)
            result = relax(candidate["smiles"], candidate["coordinates"],
                           present_mask=candidate["present_mask"], max_iterations=args.iterations)
            output = save_ensemble(args.output, candidate["smiles"], result)
        elif args.command == "score":
            output = _score(args)
        elif args.command == "select":
            output = _select(args)
        elif args.command == "identity":
            output = _identity(args)
        elif args.command == "audit-conditioning":
            from .artifacts import save_json
            from .audits import audit_conditioning
            output = save_json(args.output, audit_conditioning(args.method, args.source, seed=args.seed))
        elif args.command == "statistics":
            from .analysis import Contrast, run_statistics
            spec = json.loads(args.spec.read_text()) if args.spec else {}
            spec["contrasts"] = [Contrast(**item) for item in spec.get("contrasts", [])]
            run_statistics(args.input, args.output, draws=args.draws, seed=args.seed, **spec)
            output = args.output
        elif args.command == "export":
            from .reporting import export_metrics
            export_metrics(args.input, args.output, statistics=args.statistics)
            output = args.output
        else:
            output = _demo(args.output)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        app.exit(2, f"conformer-fidelity: {exc}\n")
    print(output)
    return 0
