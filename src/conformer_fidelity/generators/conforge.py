"""Native CONFORGE adapter with exact explicit-H atom mapping.

Generation uses the MEDIUM_SET_DIVERSE preset, AUTO sampling and native
optimization. CDPKit 1.3.0 exposes uncontrolled native sampling, represented by
seed=0. timeout_ms sets a cooperative generation-time limit. source may point
to an existing CDPKit Python installation.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np
from rdkit import Chem, rdBase

from ._common import coordinate_identity, fixed_slots, validate_request

VERSION = '1.3.0'
PRESET = 'MEDIUM_SET_DIVERSE'


@lru_cache(maxsize=1)
def cdpl_modules(source=None):
    if source is not None:
        sys.path.insert(0, str(Path(source).resolve()))
    try:
        import CDPL
        import CDPL.Chem as CDPChem
        import CDPL.ConfGen as ConfGen
    except ImportError as exc:
        raise ImportError(
            'CONFORGE requires CDPKit 1.3.0. Install it separately or provide its '
            'Python installation directory as source.'
        ) from exc
    if CDPL.VERSION_STRING != VERSION:
        raise RuntimeError('this adapter uses the CDPKit 1.3.0 native API/preset')
    if source is not None and not Path(CDPL.__file__).resolve().is_relative_to(Path(source).resolve()):
        raise RuntimeError('a different CDPL install is already imported in this process')
    return CDPChem, ConfGen


def _properties(settings):
    """Record native preset values, including the fragment-build subsettings."""
    output = {}
    for name in sorted(dir(type(settings))):
        if name == 'objectID' or not isinstance(getattr(type(settings), name), property):
            continue
        value = getattr(settings, name)
        output[name] = value if isinstance(value, (str, bool, int, float)) else _properties(value)
    return output


def make_generator(timeout_ms=60000, *, count=20, source=None):
    if not isinstance(timeout_ms, int) or timeout_ms < 0:
        raise ValueError('timeout_ms must be a nonnegative integer; 0 disables the native timeout')
    _, confgen = cdpl_modules(source)
    generator = confgen.ConformerGenerator()
    settings = generator.settings
    settings.assign(confgen.ConformerGeneratorSettings.MEDIUM_SET_DIVERSE)
    settings.samplingMode = confgen.ConformerSamplingMode.AUTO
    settings.genCoordsFromScratch = True
    settings.includeInputCoords = False
    settings.clearMaxNumOutputConformersRanges()
    settings.maxNumOutputConformers = count
    settings.timeout = timeout_ms
    return generator


def input_graph(smiles):
    if not isinstance(smiles, str) or not smiles or any(c.isspace() for c in smiles) or '|' in smiles:
        raise ValueError('a single plain 2D SMILES is required; names/CXSMILES/coordinates are not accepted')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('invalid or disconnected input graph')
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError('radical input is outside the current common-evaluator scope')
    mol = Chem.AddHs(mol)
    if mol.GetNumConformers():
        raise ValueError('input unexpectedly contains coordinates')
    return mol


def prepare_mapped_graph(molecule, *, source=None):
    """All atoms, including explicit H, get transport IDs; user maps stay intact."""
    cdchem, confgen = cdpl_modules(source)
    mapped = Chem.Mol(molecule)
    for atom in mapped.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    mapped_smiles = Chem.MolToSmiles(mapped, canonical=False, isomericSmiles=True, allHsExplicit=True)
    native = cdchem.parseSMILES(mapped_smiles)
    confgen.prepareForConformerGeneration(native, False)
    order = validate_native_graph(molecule, native, source=source)
    return native, order, mapped_smiles


def validate_native_graph(molecule, native, *, source=None):
    """Exact map-based graph identity in one common RDKit aromaticity model.

    CDPL and RDKit assign different aromatic flags to some carbonyl heterocycles.
    Compare CDPL's explicit Kekule graph after RDKit sanitization, not raw flags.
    This changes only the comparison object, never the native generator input.
    """
    cdchem, _ = cdpl_modules(source)
    order = [int(cdchem.getAtomMappingID(atom)) - 1 for atom in native.atoms]
    if sorted(order) != list(range(molecule.GetNumAtoms())):
        raise ValueError('native explicit-H atom-map identity mismatch')
    for atom, rd_index in zip(native.atoms, order):
        rd_atom = molecule.GetAtomWithIdx(rd_index)
        expected = (rd_atom.GetAtomicNum(), rd_atom.GetFormalCharge(), rd_atom.GetIsotope())
        actual = (int(cdchem.getType(atom)), int(cdchem.getFormalCharge(atom)), int(cdchem.getIsotope(atom)))
        if actual != expected:
            raise ValueError('native element/charge/isotope mismatch')
    rebuilt = Chem.RWMol()
    for atom in molecule.GetAtoms():
        copy = Chem.Atom(atom.GetAtomicNum())
        copy.SetFormalCharge(atom.GetFormalCharge())
        copy.SetIsotope(atom.GetIsotope())
        copy.SetNoImplicit(True)  # The transport graph contains every explicit H.
        rebuilt.AddAtom(copy)
    orders = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for bond in native.bonds:
        pair = tuple(sorted((order[native.getAtomIndex(bond.begin)], order[native.getAtomIndex(bond.end)])))
        bond_order = int(cdchem.getOrder(bond))
        if bond_order not in orders:
            raise ValueError('unsupported native Kekule bond order')
        rebuilt.AddBond(*pair, orders[bond_order])
    normalized = rebuilt.GetMol()
    Chem.SanitizeMol(normalized)
    bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetBondTypeAsDouble()
             for b in normalized.GetBonds()}
    expected = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetBondTypeAsDouble()
                for b in molecule.GetBonds()}
    if bonds != expected:
        raise ValueError('native bond identity/order mismatch')
    if any(a.GetIsAromatic() != b.GetIsAromatic()
           for a, b in zip(molecule.GetAtoms(), normalized.GetAtoms())):
        raise ValueError('RDKit-normalized native aromaticity mismatch')
    return np.asarray(order, dtype=np.int64)


def generate(smiles, *, count=20, seed=0, timeout_ms=60000, source=None):
    """Generate a conformer pool and record graph-to-output latency.

    Runtime includes input parsing, native setup/search, exact mapping and stereo
    validation. Module import/I/O are outside this function. The first invocation
    also includes loading the optional CDPL extension (reported separately).
    ``rows.identity_passed`` records finiteness and specified-stereo retention.
    The shared evaluator supplies bonded-distance and geometry qualification.
    """
    validate_request(count, seed)
    if seed != 0:
        raise ValueError('CONFORGE 1.3.0 has no seed setter; only seed=0 (uncontrolled) is accepted')
    started = time.perf_counter()
    import_started = time.perf_counter()
    _, confgen = cdpl_modules(source)
    import_seconds = time.perf_counter() - import_started
    generator = make_generator(timeout_ms, count=count, source=source)
    settings = _properties(generator.settings)
    result = dict(smiles=smiles, version=VERSION, rdkit=rdBase.rdkitVersion, preset=PRESET,
        native_settings=settings, native_threads=1, public_seed_parameter=None, seed_controlled=False, seed=None,
        maximum_output_count=count, native_output_count=0, native_status=None, native_status_name=None,
        deadline_observed=False, cooperative_timeout_ms=timeout_ms, failure=None,
        cdpl_import_seconds=import_seconds, native_generation_seconds=0.,
        atom_order='RDKit AddHs(MolFromSmiles(input)); all atoms including explicit H',
        target_access=False, external_optimization=False, external_selection=False,
        coordinates=np.empty((0, 0, 3)), native_energies=[], rows=[],
        native_to_rdkit=[], molecule=None)
    try:
        molecule = input_graph(smiles)
        result['molecule'] = molecule
        result['coordinates'] = np.empty((0, molecule.GetNumAtoms(), 3))
        native, order, mapped_smiles = prepare_mapped_graph(molecule, source=source)
        result.update(native_to_rdkit=order.tolist(), mapped_2d_smiles=mapped_smiles,
                      original_atom_maps=[a.GetAtomMapNum() for a in molecule.GetAtoms()])
        logs = []
        generator.setLogMessageCallback(logs.append)
        generation_started = time.perf_counter()

        def deadline():
            expired = bool(timeout_ms and (time.perf_counter() - generation_started) * 1000 >= timeout_ms)
            result['deadline_observed'] |= expired
            return expired

        generator.setTimeoutCallback(deadline)
        status = int(generator.generate(native))
        generation_seconds = time.perf_counter() - generation_started
        result.update(native_status=status, native_generation_seconds=generation_seconds,
                      native_output_count=int(generator.numConformers), native_log=logs)
        result['deadline_observed'] |= bool(timeout_ms and generation_seconds * 1000 >= timeout_ms)
        statuses = {int(getattr(confgen.ReturnCode, n)): n for n in dir(confgen.ReturnCode) if n.isupper()}
        result['native_status_name'] = statuses.get(status, str(status))
        if status not in (confgen.ReturnCode.SUCCESS, confgen.ReturnCode.TOO_MUCH_SYMMETRY):
            result['failure'] = 'native_' + result['native_status_name']
        else:
            if generator.numConformers > count:
                raise ValueError('native generator exceeded the requested output cap')
            xyz = np.empty((generator.numConformers, molecule.GetNumAtoms(), 3), dtype=np.float64)
            energies = []
            for i in range(generator.numConformers):
                conf = generator.getConformer(i)
                xyz[i, order] = np.asarray([v.toArray() for v in conf])
                energies.append(float(conf.energy))
            result.update(coordinates=xyz, native_energies=energies,
                          rows=coordinate_identity(molecule, xyz))
    except (RuntimeError, ValueError) as exc:
        result['failure'] = type(exc).__name__ + ': ' + str(exc)
    result['output_count'] = len(result['coordinates'])
    result['identity_passed_count'] = sum(row['identity_passed'] for row in result['rows'])
    result['missing_output_slots'] = count - result['output_count']
    result['graph_to_output_seconds'] = time.perf_counter() - started
    return fixed_slots(result, count)
