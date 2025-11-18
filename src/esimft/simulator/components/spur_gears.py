import openmdao.api as om
import numpy as np

class SpurGears(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('place_coord_increm', types=int)
        self.options.declare('place_sign', types=int)
        self.options.declare('r_in', types=float)
        self.options.declare('r_out', types=float)
        self.options.declare('w_in', types=float)
        self.options.declare('w_out', types=float)
        self.options.declare('N_in', types=int)
        self.options.declare('N_out', types=int)

    def setup(self):
        self.add_input('pos_in', shape=3, desc='input position', units='m')
        self.add_input('flow_dir_in', shape=3, desc='input motion flow direction')
        self.add_input('motion_vector_in', shape=3, desc='rotation/translation vector')
        self.add_input('q_dot_in', shape=1, desc='input speed', units='rad/s')

        self.add_output('pos_out', shape=3, desc='output position', units='m')
        self.add_output('flow_dir_out', shape=3, desc='output motion flow  direction')
        self.add_output('motion_vector_out', shape=3, desc='rotation/translation vector')
        self.add_output('q_dot_out', shape=1, desc='output speed', units='rad/s')

        self.add_output('bb_min', shape=3, desc='min of bounding box', units='m')
        self.add_output('bb_max', shape=3, desc='max of bounding box', units='m')

    def setup_partials(self):
        self.declare_partials('pos_out', 'pos_in', method='fd')
        self.declare_partials('pos_out', 'flow_dir_in', method='fd')
        self.declare_partials('flow_dir_out', 'flow_dir_in', method='fd')
        self.declare_partials('motion_vector_out', 'motion_vector_in', method='fd')
        self.declare_partials('q_dot_out', 'q_dot_in', method='fd')

    def compute(self, inputs, outputs):

        r_in = self.options['r_in']
        r_out = self.options['r_out']
        w_in = self.options['w_in']
        w_out = self.options['w_out']
        N_in = self.options['N_in']
        N_out = self.options['N_out']
        place_coord_increm = self.options['place_coord_increm']
        place_sign = self.options['place_sign']

        pos_in = inputs['pos_in']
        flow_dir_in = inputs['flow_dir_in']
        motion_vector_in = inputs['motion_vector_in']
        q_dot_in = inputs['q_dot_in']

        place_dir = place_sign * np.roll(np.abs(flow_dir_in), place_coord_increm)

        pos_out = pos_in + (r_in + r_out) * place_dir
        outputs['pos_out'] = pos_out
        outputs['flow_dir_out'] = flow_dir_in # will be ignored by next component
        outputs['motion_vector_out'] = -1 * motion_vector_in
        outputs['q_dot_out'] = (q_dot_in * N_in) / N_out

        ref_axis = np.abs(motion_vector_in)
        orthos = np.ones(3) - ref_axis
        bb_in_min = pos_in - r_in * orthos - 0.5 * w_in * ref_axis
        bb_in_max = pos_in + r_in * orthos + 0.5 * w_in * ref_axis
        bb_out_min = pos_out - r_out * orthos - 0.5 * w_out * ref_axis
        bb_out_max = pos_out + r_out * orthos + 0.5 * w_out * ref_axis

        outputs['bb_min'] = np.min([bb_in_min, bb_out_min], axis=0)
        outputs['bb_max'] = np.max([bb_in_max, bb_out_max], axis=0)
