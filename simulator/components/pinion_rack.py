import openmdao.api as om
import numpy as np

class PinionRack(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('place_coord_increm', types=int)
        self.options.declare('place_sign', types=int)
        self.options.declare('r_pinion', types=float)
        self.options.declare('N_pinion', types=int)
        self.options.declare('w_pinion', types=float)
        self.options.declare('N_rack', types=int)
        self.options.declare('l_rack', types=float)
        self.options.declare('h_rack', types=float)
        self.options.declare('w_rack', types=float)


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
        self.declare_partials('pos_out', 'motion_vector_in', method='fd')
        self.declare_partials('flow_dir_out', 'flow_dir_in', method='fd')
        self.declare_partials('flow_dir_out', 'motion_vector_in', method='fd')
        self.declare_partials('motion_vector_out', 'flow_dir_in', method='fd')
        self.declare_partials('motion_vector_out', 'motion_vector_in', method='fd')
        self.declare_partials('q_dot_out', 'q_dot_in', method='fd')

    def compute(self, inputs, outputs):

        r_pinion = self.options['r_pinion']
        N_pinion = self.options['N_pinion']
        w_pinion = self.options['w_pinion']
        N_rack = self.options['N_rack']
        l_rack = self.options['l_rack']
        h_rack = self.options['h_rack']
        w_rack = self.options['w_rack']
        place_coord_increm = self.options['place_coord_increm']
        place_sign = self.options['place_sign']

        pos_in = inputs['pos_in']
        flow_dir_in = inputs['flow_dir_in']
        motion_vector_in = inputs['motion_vector_in']
        q_dot_in = inputs['q_dot_in']

        place_dir = place_sign * np.roll(np.abs(flow_dir_in), place_coord_increm)
        flow_dir = np.cross(place_dir, motion_vector_in)

        pos_out = pos_in + (r_pinion + h_rack/2) * place_dir + l_rack * flow_dir
        outputs['pos_out'] = pos_out
        outputs['flow_dir_out'] = flow_dir
        outputs['motion_vector_out'] = np.cross(place_dir, motion_vector_in)
        outputs['q_dot_out'] = (q_dot_in * N_pinion) / N_rack

        pinion_axis = np.abs(motion_vector_in)
        pinion_orthos = np.ones(3) - pinion_axis
        bb_in_min = pos_in - r_pinion * pinion_orthos - 0.5 * w_pinion * pinion_axis
        bb_in_max = pos_in + r_pinion * pinion_orthos + 0.5 * w_pinion * pinion_axis

        rack_axis = np.abs(outputs['motion_vector_out'])
        rack_h_axis = np.abs(place_dir)
        rack_w_axis = np.ones(3) - rack_axis - rack_h_axis
        bb_out_min = pos_out - 0.5 * h_rack * rack_h_axis - 0.5 * w_rack * rack_w_axis
        bb_out_max = pos_out + 0.5 * h_rack * rack_h_axis + 0.5 * w_rack * rack_w_axis

        outputs['bb_min'] = np.min([bb_in_min, bb_out_min], axis=0)
        outputs['bb_max'] = np.max([bb_in_max, bb_out_max], axis=0)

