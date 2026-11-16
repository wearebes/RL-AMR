"""Local extension of the author's unchanged FSI2 entry and solver."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

CASE_DIR = Path(__file__).resolve().parent
AUTHOR_SOURCE = CASE_DIR.parents[1]/'kratos_fsi2/Examples/fluid_structure_interaction/validation/fsi_turek_FSI2/source'
sys.path.insert(0, str(AUTHOR_SOURCE))

import KratosMultiphysics as K
from KratosMultiphysics.FSIApplication.fsi_analysis import FsiAnalysis
from MainKratos import CreateAnalysisStageWithFlushInstance
from mesh_smoothing import MeshSmoothing


class SmoothedFsiAnalysis(FsiAnalysis):
    def __init__(self, model, parameters):
        self.started = time.perf_counter()
        self.smoothing = None
        self.smoothing_settings = json.loads((CASE_DIR/'mesh_smoothing_settings.json').read_text())
        self.smoothing_settings['library_path'] = str(CASE_DIR/self.smoothing_settings['library_path'])
        self.smoothing_settings['model_path'] = str(CASE_DIR/self.smoothing_settings['model_path'])
        super().__init__(model, parameters)

    def Initialize(self):
        super().Initialize()
        self.smoothing = MeshSmoothing(self._GetSolver().mesh_solver, self.smoothing_settings)

    def OutputSolutionStep(self):
        super().OutputSolutionStep()
        self.smoothing.record_step()

    def CloseSmoothing(self, completed=False):
        if self.smoothing is not None:
            self.smoothing.close(time.perf_counter()-self.started, completed)

    def Finalize(self):
        super().Finalize()
        self.CloseSmoothing(self.time >= self.end_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['o', 'prcg', 'smart_laplacian', 'vasonet'], required=True)
    parser.add_argument('--mesh-motion', choices=['structural_similarity', 'laplacian'], default='structural_similarity')
    parser.add_argument('--end-time', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parameters = K.Parameters((CASE_DIR/'ProjectParameters.json').read_text())
    parameters['solver_settings']['mesh_solver_settings']['solver_type'].SetString(args.mesh_motion)
    if args.end_time is not None:
        if args.end_time <= parameters['problem_data']['start_time'].GetDouble():
            parser.error('--end-time must be greater than start_time')
        parameters['problem_data']['end_time'].SetDouble(args.end_time)
    for solver_name in ('structure_solver_settings', 'fluid_solver_settings'):
        solver = parameters['solver_settings'][solver_name]
        mesh = solver['model_import_settings']['input_filename']
        mesh.SetString(str(AUTHOR_SOURCE/mesh.GetString()))
        material = solver['material_import_settings']['materials_filename']
        material.SetString(str(AUTHOR_SOURCE/material.GetString()))
    os.chdir(args.output.resolve())
    model = K.Model()
    simulation = CreateAnalysisStageWithFlushInstance(SmoothedFsiAnalysis, model, parameters)
    simulation.smoothing_settings['enable_smoothing'] = args.method != 'o'
    simulation.smoothing_settings['method'] = 'prcg' if args.method == 'o' else args.method
    Path('run_settings.json').write_text(json.dumps(dict(
        mesh_motion=args.mesh_motion, smoothing=args.method,
        parameters=json.loads(parameters.WriteJsonString()),
        smoothing_settings=simulation.smoothing_settings), indent=4)+'\n')
    try:
        simulation.Run()
    finally:
        simulation.CloseSmoothing()
        # Release the strategy's raw mesh-part pointer before destroying its Model.
        mesh_solver = simulation._GetSolver().mesh_solver
        if hasattr(mesh_solver, '_mesh_motion_solving_strategy'):
            del mesh_solver._mesh_motion_solving_strategy
        del simulation
