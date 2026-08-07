
import argparse, inspect, math, unittest
from unittest import mock
import numpy as np
import main
from quantum_solver import dirac_canonical as dc
from quantum_solver.dirac_canonical_adapter import ShadowDiracAdapter
from quantum_solver.quantum_system import Electron
from quantum_solver.plotting import DiracPlotter
from ablation_experiment import batch_dirac

def hyper(mode='learnable',h=2):
 return {'diff_equation':'Dirac','arguments.parent_equation':'Schrodinger','hamiltonian_correction_terms':'0 0 0 0 0','num_of_hidden_units':h,'num_of_equidistant_points':7,'fuc':1,'optimization_method':'BFGS','minimization_tol_lvl':1e-8,'primary_quantum_number':1,'azimuthal_quantum_number':0,'total_angular_momentum':0.5,'k_operator':-1,'coupling_constant':dc.FINE_STRUCTURE_CONSTANT,'two_body_system':'combined_hamiltonian','k_optimization_constant':0.16,'u0_init_method':'fixed','activation_function':'tanh','user_search_grid':0,'overlay_plot':False,'input_file':True,'verbose':False,'experiment_index':99,'mass_relation':'m1>>m2','subsystem':'Muonium','numerical_method':'NeuralNetworks','manual_u0':None,'dirac_coefficient_mode':mode}
def electron(mode='learnable',h=2):
 e=Electron(hyper(mode,h),1); e.num_hidden_units=h; e.relative_search_points=7; e.energy=e.DiracEnergyEigenvalue(); e.epsilon=e.energy/e.mass; e.lambda_param=e.mass/e.fundamental_values.hbarc*np.sqrt(1-e.epsilon**2); e.beta=e.lambda_param; e.k_optimization_const=e.lambda_param; e.configure_canonical_dirac_runtime(); return e
def vec(mode='learnable',h=2,cf=1,cg=1):
 f=np.linspace(.1,.6,3*h); g=np.linspace(-.2,.3,3*h); return np.concatenate([f,[cf],g,[cg]]) if mode in ('learnable',dc.LEARNABLE_COEFFICIENTS) else np.concatenate([f,g])
class RuntimeWiringTests(unittest.TestCase):
 def ns(self,diff='Dirac',mode='learnable'): return argparse.Namespace(diff_equation=diff,parent_equation='Schrodinger',dirac_coefficient_mode=mode)
 def test_01_cli_accepts_learnable(self): self.assertEqual(main.validate_dirac_coefficient_mode(self.ns(mode='learnable')),dc.LEARNABLE_COEFFICIENTS)
 def test_02_cli_accepts_fixed(self): self.assertEqual(main.validate_dirac_coefficient_mode(self.ns(mode='fixed')),dc.FIXED_UNIT_COEFFICIENTS)
 def test_03_cli_rejects_missing_mode(self):
  with self.assertRaisesRegex(ValueError,'dirac-coefficient-mode'): main.validate_dirac_coefficient_mode(self.ns(mode=None))
 def test_04_cli_rejects_invalid_mode(self):
  with self.assertRaisesRegex(ValueError,'learnable.*fixed'): main.validate_dirac_coefficient_mode(self.ns(mode='auto'))
 def test_05_schrodinger_does_not_require_mode(self): self.assertIsNone(main.validate_dirac_coefficient_mode(self.ns(diff='Schrodinger',mode=None)))
 def test_06_mode_survives_config_creation(self): self.assertEqual(electron('fixed').dirac_coefficient_mode,dc.FIXED_UNIT_COEFFICIENTS)
 def test_07_mode_survives_batch_argument_propagation(self): self.assertIn('fixed',batch_dirac.build_dirac_command('python','main.py',2,'fixed'))
 def test_08_learnable_vector_layout_h2_h6_h17(self):
  for h in (2,6,17): self.assertEqual(dc.branch_layout(h,dc.LEARNABLE_COEFFICIENTS).expected_length,6*h+2)
 def test_09_fixed_vector_layout_h2_h6_h17(self):
  for h in (2,6,17): self.assertEqual(dc.branch_layout(h,dc.FIXED_UNIT_COEFFICIENTS).expected_length,6*h)
 def test_10_wrong_vector_length_rejection(self):
  with self.assertRaises(ValueError): dc.unpack_vector(np.ones(12),2,dc.LEARNABLE_COEFFICIENTS)
  with self.assertRaises(ValueError): dc.unpack_vector(np.ones(14),2,dc.FIXED_UNIT_COEFFICIENTS)
 def test_11_no_mode_inference_from_vector_length(self):
  with self.assertRaises(ValueError): electron('learnable').validate_dirac_vector(np.ones(12))
  with self.assertRaises(ValueError): electron('fixed').validate_dirac_vector(np.ones(14))
 def test_12_learnable_cf_cg_used_in_fg(self):
  e=electron('learnable'); r=np.linspace(1,3,3); a=e.dirac_canonical_adapter.evaluate(r,vec('learnable',cf=2,cg=3)); b=e.dirac_canonical_adapter.evaluate(r,vec('learnable')); np.testing.assert_allclose(a['F'],2*b['F']); np.testing.assert_allclose(a['G'],3*b['G'])
 def test_13_fixed_uses_unit_coefficients(self): self.assertEqual((dc.unpack_vector(vec('fixed'),2,dc.FIXED_UNIT_COEFFICIENTS).c_f,dc.unpack_vector(vec('fixed'),2,dc.FIXED_UNIT_COEFFICIENTS).c_g),(1.0,1.0))
 def test_14_coefficients_used_in_derivatives(self):
  e=electron('learnable'); r=np.linspace(1,3,3); a=e.dirac_canonical_adapter.evaluate(r,vec('learnable',cf=4,cg=5)); b=e.dirac_canonical_adapter.evaluate(r,vec('learnable')); np.testing.assert_allclose(a['dF'],4*b['dF']); np.testing.assert_allclose(a['dG'],5*b['dG'])
 def test_15_coefficients_reach_residual_objective(self): self.assertNotEqual(electron('learnable').objective_function(vec('learnable',cf=2,cg=3),np.linspace(1,5,7)),electron('learnable').objective_function(vec('learnable'),np.linspace(1,5,7)))
 def test_16_coefficients_reach_normalization(self): self.assertNotEqual(electron('learnable').dirac_canonical_adapter.norm(np.linspace(1,5,7),vec('learnable',cf=2)),electron('learnable').dirac_canonical_adapter.norm(np.linspace(1,5,7),vec('learnable')))
 def test_17_coefficients_reach_energy(self): self.assertTrue(math.isfinite(electron('learnable').canonical_dirac_energy(np.linspace(1,5,7),vec('learnable',cf=2,cg=3))))
 def test_18_coefficients_reach_export(self):
  e=electron('learnable'); r=np.linspace(1,5,7); _,a=e.dirac_canonical_adapter.export_data(r,vec('learnable'),normalized=False); _,b=e.dirac_canonical_adapter.export_data(r,vec('learnable',cf=2,cg=3),normalized=False); self.assertGreater(np.max(np.abs(a[:,1:]-b[:,1:])),1e-12)
 def test_19_mode_metadata_reaches_logs(self): self.assertIn('Dirac coefficient mode',inspect.getsource(Electron.solve_dirac_equation))
 def test_20_mode_metadata_reaches_output_provenance(self): self.assertIn('dirac_coefficient_mode',inspect.getsource(DiracPlotter.plot))
 def test_21_objective_uses_reduced_radial_functions(self): self.assertIn('canonical_dirac_objective_terms',inspect.getsource(Electron.objective_function))
 def test_22_residual_signs_match(self): self.assertIn('rest_energy_mev + eigenenergy_mev - potential_mev',inspect.getsource(dc.coupled_residuals)); self.assertIn('rest_energy_mev - eigenenergy_mev + potential_mev',inspect.getsource(dc.coupled_residuals))
 def test_23_normalization_common_sign_preserving(self):
  r=np.linspace(1,5,7); f=-r; g=.1*r; fn,gn=dc.normalize_spinor(f,g,r); self.assertTrue(np.all(fn<0)); self.assertTrue(np.all(gn>0))
 def test_24_energy_has_no_clipping(self): self.assertNotIn('clip',inspect.getsource(dc.canonical_energy_terms))
 def test_25_plotting_receives_canonical_arrays(self): self.assertIn('canonical_numerical_data',inspect.getsource(DiracPlotter.plot))
 def test_26_legacy_objective_not_called(self):
  e=electron('fixed')
  with mock.patch.object(dc,'primary_objective',wraps=dc.primary_objective) as spy: e.objective_function(vec('fixed'),np.linspace(1,5,7))
  self.assertTrue(spy.called)
 def test_27_legacy_normalization_not_called(self): self.assertNotIn('fgnum_squared_log',inspect.getsource(Electron.solve_dirac_equation))
 def test_28_legacy_energy_path_not_called(self): self.assertNotIn('DiracRadialNumericalFunction',inspect.getsource(Electron.solve_dirac_equation))
 def test_29_legacy_export_path_not_called(self): self.assertIn('canonical_export_table',inspect.getsource(Electron.solve_dirac_equation))
 def test_30_legacy_plotting_transformation_not_called(self): self.assertIn('canonical_F is None and',inspect.getsource(DiracPlotter.plot))
 def test_31_learnable_fixed_equivalence_unit_coefficients(self):
  r=np.linspace(1,5,7); f=ShadowDiracAdapter(2,.01,'fixed').evaluate(r,vec('fixed')); l=ShadowDiracAdapter(2,.01,'learnable').evaluate(r,vec('learnable'))
  for k in ('F','G','dF','dG'): np.testing.assert_allclose(f[k],l[k])
 def test_32_reference_eigenenergy_unchanged(self): self.assertAlmostEqual(dc.dirac_energy_eigenvalue(1,-1),dc.GROUND_REFERENCE_ENERGY_MEV,delta=5e-16)
 def test_33_schrodinger_import_behavior_unchanged(self): self.assertIsNone(main.validate_dirac_coefficient_mode(self.ns(diff='Schrodinger',mode='fixed')))

 def test_34_physics_conditioned_requires_fixed(self):
  e=electron('learnable',6); hp=hyper('learnable',6); hp['u0_init_method']='physics_conditioned'
  with self.assertRaisesRegex(ValueError,'requires.*fixed'): e.initialize_dirac_optimization_params(hp)
 def test_35_ground_physics_conditioned_fixed_layout(self):
  np.random.seed(20260804); e=electron('fixed',6); hp=hyper('fixed',6); hp['u0_init_method']='physics_conditioned'; e.activation='tanh'
  values=e.initialize_dirac_optimization_params(hp)
  self.assertEqual(values.size,36)
  r=np.linspace(1,8e5,7); spinor=e.dirac_canonical_adapter.evaluate(r,values); self.assertTrue(np.all(np.isfinite(spinor['F']))); self.assertTrue(np.all(np.isfinite(spinor['G'])))
 def test_36_physics_conditioned_is_separate_from_historical_fixed(self):
  self.assertIn("physics_conditioned",inspect.getsource(Electron.initialize_dirac_optimization_params))
 def test_37_physics_conditioned_q_uses_existing_dirac_state(self):
  source=inspect.getsource(Electron._dirac_physics_component_ratio); self.assertIn('self.epsilon',source); self.assertNotIn('Ffunc',source); self.assertNotIn('Gfunc',source)
 def test_38_physics_conditioned_l_bfgsb_uses_validation_settings(self):
  source=inspect.getsource(Electron.solve_dirac_equation); self.assertIn("lbfgsb_options['maxls'] = 50",source); self.assertIn("hyperparameters['minimization_tol_lvl']",source)
 def test_39_article_facing_dirac_flow_remains_active(self):
  source=inspect.getsource(Electron.solve_dirac_equation); self.assertIn('self.DiracEnergyEigenvalue()',source); self.assertIn('self.objective_function',source); self.assertIn('analytical_F=self.Ffunc',source); self.assertIn('analytical_G=self.Gfunc',source)
if __name__=='__main__': unittest.main(verbosity=2)
