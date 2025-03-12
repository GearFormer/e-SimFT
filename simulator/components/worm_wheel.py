import openmdao.api as om
import numpy as np

class WormWheel(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('place_coord_increm', types=int)
        self.options.declare('place_sign', types=int)
        self.options.declare('N_worm', types=int)
        self.options.declare('N_wheel', types=int)
        self.options.declare('l_worm', types=float)
        self.options.declare('r_worm', types=float)
        self.options.declare('r_wheel', types=float)
        self.options.declare('l_wheel', types=float)

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
        self.declare_partials('motion_vector_out', 'flow_dir_in', method='fd')
        self.declare_partials('motion_vector_out', 'motion_vector_in', method='fd')
        self.declare_partials('q_dot_out', 'q_dot_in', method='fd')

    def compute(self, inputs, outputs):

        l_worm = self.options['l_worm']
        r_worm = self.options['r_worm']
        r_wheel = self.options['r_wheel']
        place_coord_increm = self.options['place_coord_increm']
        place_sign = self.options['place_sign']
        N_worm = self.options['N_worm']
        N_wheel = self.options['N_wheel']
        l_wheel = self.options['l_wheel']

        pos_in = inputs['pos_in']
        flow_dir_in = inputs['flow_dir_in']
        motion_vector_in = inputs['motion_vector_in']
        q_dot_in = inputs['q_dot_in']

        place_dir = place_sign * np.roll(np.abs(flow_dir_in), place_coord_increm)

        pos_out = pos_in + (l_worm/2) * flow_dir_in + (r_worm + r_wheel) * place_dir
        outputs['pos_out'] = pos_out
        outputs['flow_dir_out'] = np.cross(place_dir, flow_dir_in) # the sign of rot_coord_dir does not matter
        outputs['motion_vector_out'] = -1 * np.cross(place_dir, motion_vector_in)
        outputs['q_dot_out'] = (q_dot_in * N_worm) / N_wheel

        worm_axis = np.abs(motion_vector_in)
        worm_orthos = np.ones(3) - worm_axis
        bb_in_min = pos_in - r_worm * worm_orthos - 0.5 * l_worm * worm_axis
        bb_in_max = pos_in + r_worm * worm_orthos + 0.5 * l_worm * worm_axis

        wheel_axis = np.abs(outputs['motion_vector_out'])
        wheel_orthos = np.ones(3) - wheel_axis
        bb_out_min = pos_out - r_wheel * wheel_orthos - 0.5 * l_wheel * wheel_axis
        bb_out_max = pos_out + r_wheel * wheel_orthos + 0.5 * l_wheel * wheel_axis

        outputs['bb_min'] = np.min([bb_in_min, bb_out_min], axis=0)
        outputs['bb_max'] = np.max([bb_in_max, bb_out_max], axis=0)

