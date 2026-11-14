"""FSI2 mesh smoothing: Mesquite PR-CG, Smart Laplacian and dual MLP inference."""
import csv
import ctypes as ct
import json
from pathlib import Path
import time
import weakref

import numpy as np
import KratosMultiphysics as K


class NativeResult(ct.Structure):
    _fields_ = [(name, ct.c_int32) for name in ('status', 'called', 'selected', 'moved')] + [
        ('seconds', ct.c_double)]


def quality(points, faces):
    a, b, c = points[faces, :2].transpose(1, 0, 2)
    u, v = b-a, c-a
    det = u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]
    denom = (u*u).sum(1)+(v*v).sum(1)+((v-u)**2).sum(1)
    return det, 2*np.sqrt(3)*det/denom


class PatchSmoother:
    def __init__(self, native, settings, triangles, count):
        self.native = native
        self.method = settings['method']
        self.incident = [[] for _ in range(count)]
        for face, vertices in enumerate(triangles):
            for vertex in vertices:
                self.incident[vertex].append(face)
        self.rings = []
        for vertex, incident in enumerate(self.incident):
            # Recover the oriented topological cycle; angle sorting can hide holes.
            links = {}
            for face in triangles[incident]:
                k = list(face).index(vertex)
                links[int(face[(k+1)%3])] = int(face[(k+2)%3])
            ring = []
            if links:
                current = min(links)
                while current not in ring and current in links:
                    ring.append(current)
                    current = links[current]
                if current != ring[0] or len(ring) != len(incident):
                    ring = []
            self.rings.append(np.asarray(ring, dtype=np.int64))
        if self.method == 'vasonet':
            solver_threads = K.ParallelUtilities.GetNumThreads()
            import torch
            # PyTorch import changes shared OpenMP state; retain the solver budget.
            torch.set_num_threads(solver_threads)
            from model import VASONet
            self.torch = torch
            self.model = VASONet().cpu().eval()
            self.model.load_state_dict(torch.load(settings['model_path'], map_location='cpu', weights_only=True))
            self.model.double()

    def smooth(self, coordinates, triangles, fixed, settings):
        started = time.perf_counter()
        result = NativeResult(called=1)
        # Native measurement rejects invalid input before any proposal is attempted.
        q, _, _ = self.native.measure(coordinates, triangles)
        candidates = np.unique(triangles[q < settings['quality_threshold']])
        candidates = candidates[fixed[candidates] == 0][:settings['max_selected_vertices']]
        result.selected = len(candidates)
        work = coordinates.copy()
        for _ in range(settings['outer_iteration_limit']):
            for vertex in candidates:
                ring_ids = self.rings[vertex]
                if len(ring_ids) < 3:
                    continue
                ring = work[ring_ids, :2]
                if self.method == 'smart_laplacian':
                    proposal = ring.mean(axis=0)
                else:
                    # Unsupported valences remain unchanged: no hidden CG fallback.
                    if len(ring) > 9:
                        continue
                    lower = ring.min(axis=0)
                    scale = np.ptp(ring, axis=0).max()
                    if scale <= 0:
                        continue
                    padded = np.zeros((9, 2), dtype=np.float64)
                    padded[:len(ring)] = (ring-lower)/scale
                    with self.torch.inference_mode():
                        prediction = self.model(self.torch.from_numpy(padded.reshape(1, 18)),
                            self.torch.tensor([[len(ring)]], dtype=self.torch.float64))[0].numpy()
                    proposal = prediction*scale+lower
                if not np.isfinite(proposal).all():
                    continue
                faces = triangles[self.incident[vertex]]
                before = quality(work, faces)[1].min()
                original = work[vertex, :2].copy()
                work[vertex, :2] = proposal
                det, after = quality(work, faces)
                if np.all(det > 0) and after.min() > before:
                    result.moved += 1
                else:
                    work[vertex, :2] = original
        # Reuse the native validator before committing the complete sweep.
        self.native.measure(work, triangles)
        coordinates[:] = work
        result.status = 1 if result.moved else 0
        result.seconds = time.perf_counter()-started
        return result, ''


class MesquiteLibrary:
    def __init__(self, path):
        self.library = ct.CDLL(str(path))
        self.library.mesquite_smoothing_abi.restype = ct.c_int
        if self.library.mesquite_smoothing_abi() != 5:
            raise RuntimeError('Incompatible Mesquite bridge ABI: the new objective requires a rebuild')
        self.library.mesquite_smoothing_version.restype = ct.c_char_p
        self.version = self.library.mesquite_smoothing_version().decode()
        if '2.3.0' not in self.version:
            raise RuntimeError(f'Expected Mesquite 2.3.0, got {self.version}')
        self.run = self.library.mesquite_smooth
        array = np.ctypeslib.ndpointer
        self.run.argtypes = [ct.c_int64, array(np.float64, ndim=2, flags='C_CONTIGUOUS'),
            ct.c_int64, array(np.int64, ndim=2, flags='C_CONTIGUOUS'),
            array(np.int32, ndim=1, flags='C_CONTIGUOUS'), ct.c_double, ct.c_int64,
            ct.c_int32, ct.c_int32, ct.c_double, ct.c_double,
            ct.POINTER(NativeResult), ct.POINTER(ct.c_char), ct.c_uint64]
        self.run.restype = ct.c_int
        self.measure_run = self.library.mesquite_measure
        self.measure_run.argtypes = [ct.c_int64, array(np.float64, ndim=2, flags='C_CONTIGUOUS'),
            ct.c_int64, array(np.int64, ndim=2, flags='C_CONTIGUOUS'),
            array(np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_char), ct.c_uint64]
        self.measure_run.restype = ct.c_int

    def measure(self, coordinates, triangles, with_jacobian=False):
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError('Expected coordinates (N, 3) and triangle connectivity (E, 3)')
        quality = np.empty(len(triangles), dtype=np.float64)
        mean, minimum = ct.c_double(), ct.c_double()
        min_j = ct.c_double()
        message = ct.create_string_buffer(4096)
        status = self.measure_run(len(coordinates), coordinates, len(triangles), triangles,
            quality, ct.byref(mean), ct.byref(minimum),
            ct.byref(min_j) if with_jacobian else None, message, len(message))
        if status:
            raise RuntimeError(message.value.decode('utf8', errors='replace'))
        result = (quality, mean.value, minimum.value)
        return (*result, min_j.value) if with_jacobian else result

    def smooth(self, coordinates, triangles, fixed, settings):
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not coordinates.flags.writeable:
            raise ValueError('Coordinates must be a writable (N, 3) array')
        if triangles.ndim != 2 or triangles.shape[1] != 3 or fixed.shape != (len(coordinates),):
            raise ValueError('Expected triangle connectivity (E, 3) and fixed flags (N,)')
        result = NativeResult()
        message = ct.create_string_buffer(4096)
        self.run(len(coordinates), coordinates, len(triangles), triangles, fixed,
            settings['quality_threshold'], settings['max_selected_vertices'],
            settings['inner_iteration_limit'], settings['outer_iteration_limit'],
            settings['regularization_delta'], settings['softmin_alpha'],
            ct.byref(result), message, len(message))
        return result, message.value.decode('utf8', errors='replace')


class MeshSmoothing:
    def __init__(self, mesh_solver, settings, output_dir='.'):
        if hasattr(mesh_solver, '_fsi2_smoothing_hook'):
            raise RuntimeError('Mesh smoothing is already installed')
        self.settings = dict(settings)
        if self.settings['method'] not in ('prcg', 'smart_laplacian', 'vasonet'):
            raise ValueError(f"Unknown smoothing method: {self.settings['method']}")
        if not 0 <= self.settings['quality_threshold'] <= 1:
            raise ValueError('quality_threshold must be between 0 and 1')
        for name in ('max_selected_vertices', 'inner_iteration_limit', 'outer_iteration_limit'):
            value = self.settings[name]
            if type(value) is not int or value < (0 if name == 'max_selected_vertices' else 1):
                raise ValueError(f'Invalid {name}')
        self.mesh_solver = mesh_solver
        self.part = mesh_solver.mesh_model_part
        if self.part.GetCommunicator().GetDataCommunicator().Size() != 1:
            raise ValueError('This FSI2 bridge supports serial/OpenMP runs, not MPI')
        if self.part.ProcessInfo[K.DOMAIN_SIZE] != 2:
            raise ValueError('This bridge supports 2D FSI2 triangles only')
        if mesh_solver.settings['superimpose_mesh_disp_with'].size() or mesh_solver.settings['superimpose_mesh_velocity_with'].size():
            raise ValueError('Superimposed mesh fields are outside this FSI2 integration')
        # Sort references only: the solver's node container and IDs are unchanged.
        # ArrayMesh handles then follow ascending real Node ID, including gaps.
        self.nodes = sorted(self.part.Nodes, key=lambda node: node.Id)
        id_to_index = {n.Id: i for i, n in enumerate(self.nodes)}
        if any(len(e.GetGeometry()) != 3 for e in self.part.Elements):
            raise ValueError('Only linear triangles are supported')
        self.triangles = np.array([[id_to_index[n.Id] for n in e.GetGeometry()]
                                  for e in self.part.Elements], dtype=np.int64)
        self.reference = np.array([[n.X0, n.Y0, n.Z0] for n in self.nodes], dtype=np.float64)
        self.coordinates = np.empty_like(self.reference)
        self.displacement = np.empty_like(self.reference)
        self.boundary_ids = {n.Id for c in self.part.Conditions for n in c.GetGeometry()}
        self.fixed = np.zeros(len(self.nodes), dtype=np.int32)
        # Measurements and validity checks use Mesquite even when PR-CG is disabled.
        self.native = MesquiteLibrary(self.settings['library_path'])
        self.optimizer = self.native
        if self.settings['enable_smoothing'] and self.settings['method'] != 'prcg':
            self.optimizer = PatchSmoother(self.native, self.settings, self.triangles, len(self.nodes))
        self.measure = self.native.measure
        self.measure(self.reference, self.triangles)
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output/'mesh_smoothing.csv').exists():
            raise FileExistsError(f'Refusing to overwrite previous smoothing records in {self.output}')
        self.file = (self.output/'mesh_smoothing.csv').open('w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=[
            'record_type','step','time','min_J','min_q','mean_q','prcg_calls','prcg_seconds',
            'smoothing_seconds','rejected_calls','reason','total_seconds','completed'])
        self.metric_prefix = 'prcg' if self.settings['method'] == 'prcg' else 'optimizer'
        self.writer.fieldnames = [name.replace('prcg_', self.metric_prefix+'_') for name in self.writer.fieldnames]
        self.writer.writeheader()
        self._row('configuration', reason=json.dumps(dict(settings=self.settings,
            kratos_version=K.KratosGlobals.Kernel.Version(),mesquite_version=self.native.version,
            threads=K.ParallelUtilities.GetNumThreads(),jacobian='2*signed_area, XY, m^2',
            optimization_objective=('regularized_mean_ratio_logsumexp' if self.settings['method'] == 'prcg' else self.settings['method']),
            reported_quality='ordinary_mean_ratio',
            candidate_order='ascending_node_id',bridge_abi=5)))
        self.calls = self.step_calls = self.rejections = self.step_rejections = 0
        self.seconds = self.step_seconds = 0.0
        self.smoothing_seconds = self.step_smoothing_seconds = 0.0
        self.original_move = mesh_solver.MoveMesh
        # Weak ownership: do not create mesh_solver -> callback -> mesh_solver cycle.
        hook = weakref.ref(self)
        def move_mesh():
            owner = hook()
            if owner is None:
                raise RuntimeError('Mesh smoothing hook was destroyed without detaching')
            return owner._move_mesh()
        mesh_solver.MoveMesh = move_mesh
        mesh_solver._fsi2_smoothing_hook = True
        self.closed = False

    def _refresh(self):
        for i, node in enumerate(self.nodes):
            self.displacement[i] = node.GetSolutionStepValue(K.MESH_DISPLACEMENT, 0)
            self.fixed[i] = (node.Id in self.boundary_ids or node.Is(K.INTERFACE)
                            or node.IsFixed(K.MESH_DISPLACEMENT_X) or node.IsFixed(K.MESH_DISPLACEMENT_Y))
        np.add(self.reference, self.displacement, out=self.coordinates)

    def _reject(self, reason):
        self.rejections += 1
        self.step_rejections += 1
        self._row('rejection', reason=reason)

    def _row(self, record_type, **values):
        values = {key.replace('prcg_', self.metric_prefix+'_'): value for key, value in values.items()}
        self.writer.writerow(dict(record_type=record_type,step=int(self.part.ProcessInfo[K.STEP]),
            time=float(self.part.ProcessInfo[K.TIME]),**values))
        self.file.flush()

    def _add_smoothing_time(self, started):
        elapsed = time.perf_counter()-started
        self.smoothing_seconds += elapsed
        self.step_smoothing_seconds += elapsed

    def _move_mesh(self):
        started = time.perf_counter()
        try:
            self._refresh()
            if self.settings['enable_smoothing']:
                self._smooth()
            else:
                self.measure(self.coordinates, self.triangles)
        finally:
            self._add_smoothing_time(started)
        # Exactly one call: native Kratos owns coordinate/velocity/acceleration updates.
        return self.original_move()

    def _smooth(self):
        result, message = self.optimizer.smooth(
            self.coordinates, self.triangles, self.fixed, self.settings)
        self.calls += result.called
        self.step_calls += result.called
        self.seconds += result.seconds
        self.step_seconds += result.seconds
        if result.status == 3:
            raise RuntimeError('Original mesh/input rejected: '+message)
        if result.status == 0:
            return
        if result.status != 1:
            self._reject(message or f'Unexpected native status {result.status}')
            return

        # The optimizer has validated geometry and preserved fixed vertices.
        changed = np.flatnonzero(np.any(self.reference+self.displacement != self.coordinates, axis=1))
        try:
            for index in changed:
                displacement = self.coordinates[index]-self.reference[index]
                self.nodes[index].SetSolutionStepValue(K.MESH_DISPLACEMENT, 0, displacement)
        except Exception as error:
            # Restore current displacement if a Kratos setter fails partway through.
            for index in changed:
                self.nodes[index].SetSolutionStepValue(K.MESH_DISPLACEMENT, 0, self.displacement[index])
            self._reject(f'{type(error).__name__}: {error}')

    def record_step(self):
        started = time.perf_counter()
        current = np.array([[n.X,n.Y,n.Z] for n in self.nodes])
        _, mean, minimum, min_j = self.measure(current, self.triangles, with_jacobian=True)
        self._add_smoothing_time(started)
        self._row('step', min_J=min_j,min_q=minimum,mean_q=mean,prcg_calls=self.step_calls,
            prcg_seconds=self.step_seconds,smoothing_seconds=self.step_smoothing_seconds,
            rejected_calls=self.step_rejections)
        self.step_calls = self.step_rejections = 0
        self.step_seconds = self.step_smoothing_seconds = 0.0

    def close(self, total_seconds, completed=False):
        if self.closed:
            return
        self.mesh_solver.MoveMesh = self.original_move
        del self.mesh_solver._fsi2_smoothing_hook
        self._row('summary',total_seconds=total_seconds,completed=bool(completed),
            prcg_calls=self.calls,prcg_seconds=self.seconds,smoothing_seconds=self.smoothing_seconds,
            rejected_calls=self.rejections)
        self.file.close()
        self.closed = True
        self.original_move = self.mesh_solver = None
