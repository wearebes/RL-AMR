"""Extract beam-tip displacement and official Kratos forces from saved GiD BIN."""
import argparse
import csv
import gzip
from pathlib import Path
import re
import struct

import numpy as np
import KratosMultiphysics as K
import KratosMultiphysics.FluidDynamicsApplication as CFD


SOURCE = Path(__file__).resolve().parents[2] / (
    'kratos_fsi2/Examples/fluid_structure_interaction/validation/fsi_turek_FSI2/source')
METHODS = ('similarity', 'laplacian', 'laplacian_prcg', 'laplacian_nn')


def read_vectors(path, variable):
    """Read this project's GiDPostEx1.2 indexed float32 nodal vectors.

    Vector records hold node ID, x/y/z and magnitude. Unsupported encodings
    fail explicitly rather than interpreting another format as these records.
    """
    data = gzip.decompress(path.read_bytes())
    if data[:4] != struct.pack('<i', 2333) or b'GiDPostEx1.2\0' not in data[:30]:
        raise ValueError(f'Unsupported GiD binary header: {path}')
    header = re.compile(rb'Result "' + variable.encode() + rb'" "Kratos" ([^ ]+) Vector OnNodes\x00')
    start_marker = struct.pack('<i', 18) + b'Values -1 Indexed\0'
    end_marker = b'\xff\xff\xff\xff' + struct.pack('<i', 11) + b'End Values\0'
    dtype = np.dtype([('id', '<i4'), ('value', '<f4', (4,))])
    results = {}
    for match in header.finditer(data):
        length = struct.unpack_from('<i', data, match.start() - 4)[0]
        if length != len(match.group()):
            raise ValueError('Invalid result header length')
        start = match.end()
        if data[start:start + len(start_marker)] != start_marker:
            raise ValueError('Expected indexed nodal values')
        start += len(start_marker)
        end = data.find(end_marker, start)
        if end < 0 or (end - start) % dtype.itemsize:
            raise ValueError('Truncated or unsupported vector records')
        records = np.frombuffer(data[start:end], dtype=dtype)
        ids, values = records['id'], records['value'].astype(np.float64)
        if (ids <= 0).any() or len(np.unique(ids)) != len(ids) or not np.isfinite(values).all():
            raise ValueError('Invalid node IDs or vector values')
        if not np.allclose(np.linalg.norm(values[:, :3], axis=1), values[:, 3],
                           rtol=2e-6, atol=1e-12):
            raise ValueError('Vector magnitude check failed')
        time = float(match.group(1))
        if time in results:
            raise ValueError('Duplicate result time')
        results[time] = (ids.copy(), values[:, :3])
    if not results:
        raise ValueError(f'No {variable} nodal vectors in {path}')
    return results


def nodes(text):
    block = text.split('Begin Nodes', 1)[1].split('End Nodes', 1)[0]
    return {int(row[0]): np.array(list(map(float, row[1:])))
            for line in block.splitlines() if (row := line.split())}


def obstacle(text):
    ids = set()
    for name in ('NoSlip2D_Cylinder', 'FluidNoSlipInterface2D_FluidInterface'):
        block = text.split('Begin SubModelPart ' + name, 1)[1]
        block = block.split('Begin SubModelPartNodes', 1)[1].split('End SubModelPartNodes', 1)[0]
        ids.update(map(int, block.split()))
    return sorted(ids)


def tip_weights(text):
    """Evaluate A=(0.6,0.2) with the structural boundary's linear FE trace."""
    edge = sorted((xyz[1], node) for node, xyz in nodes(text).items()
                  if np.isclose(xyz[0], 0.6, rtol=0, atol=1e-10))
    for y, node in edge:
        if abs(y - 0.2) < 1e-10:
            return [node], np.array([1.0])
    lower = max((y, node) for y, node in edge if y < 0.2)
    upper = min((y, node) for y, node in edge if y > 0.2)
    pair = [lower[1], upper[1]]
    connected = False
    for block in re.findall(r'Begin Elements[^\n]*\n(.*?)End Elements', text, re.S):
        for line in block.splitlines():
            ids = list(map(int, line.split()))[2:]
            if pair[0] in ids and pair[1] in ids:
                i, j = ids.index(pair[0]), ids.index(pair[1])
                connected |= (i - j) % len(ids) in (1, len(ids) - 1)
    if not connected:
        raise ValueError('Tip interpolation nodes must form a structural element edge')
    fraction = (0.2 - lower[0]) / (upper[0] - lower[0])
    return pair, np.array([1 - fraction, fraction])


def extract(directory):
    fluid_text = (SOURCE / 'fsi_turek_FSI2_Fluid.mdpa').read_text()
    structural_text = (SOURCE / 'fsi_turek_FSI2_Structural.mdpa').read_text()
    boundary_ids = obstacle(fluid_text)
    tip_ids, weights = tip_weights(structural_text)
    reactions = read_vectors(directory / 'fsi_turek_FSI2_fluid.post.bin', 'REACTION')
    displacement = read_vectors(directory / 'fsi_turek_FSI2_structure.post.bin', 'DISPLACEMENT')
    if reactions.keys() != displacement.keys():
        raise ValueError('Fluid and structure saved times differ')
    model = K.Model()
    boundary = model.CreateModelPart('Obstacle')
    boundary.AddNodalSolutionStepVariable(K.REACTION)
    coordinates = nodes(fluid_text)
    for node_id in boundary_ids:
        boundary.CreateNewNode(node_id, *coordinates[node_id])
    utility = CFD.FlowForcesAndMomentsUtilities()
    rows = []
    for time in sorted(reactions):
        ids, values = reactions[time]
        lookup = {int(node): index for index, node in enumerate(ids)}
        selected = values[[lookup[node] for node in boundary_ids]]
        for node_id, value in zip(boundary_ids, selected):
            boundary.GetNode(node_id).SetSolutionStepValue(K.REACTION, value.tolist())
        force, _ = utility.CalculateBodyFittedFlowForcesAndMoments(boundary, K.Array3([0, 0, 0]))
        # Independent sum verifies mapping, sign and the official utility result.
        if not np.allclose(list(force), -selected.sum(axis=0), rtol=1e-10, atol=1e-9):
            raise ValueError('Official force and independent reaction sum disagree')
        ids, values = displacement[time]
        lookup = {int(node): index for index, node in enumerate(ids)}
        tip = weights @ values[[lookup[node] for node in tip_ids]]
        rows.append([time, tip[0], tip[1], force[0], force[1]])
    with (directory / 'evaluation.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time', 'tip_dx', 'tip_dy', 'drag', 'lift'])
        writer.writerows(rows)
    print(f'{directory.name}: {len(rows)} times, {len(boundary_ids)} obstacle nodes; '
          f'A interpolation nodes={tip_ids}, weights={weights.tolist()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    for name in METHODS:
        directory = args.root / name
        path = directory / 'mesh_smoothing.csv'
        if not path.exists():
            continue
        with path.open(newline='') as file:
            summary = [row for row in csv.DictReader(file) if row['record_type'] == 'summary']
        if not summary or summary[-1]['completed'].lower() != 'true':
            print(f'{name}: still incomplete; leave BIN untouched')
            continue
        extract(directory)
