import openmdao.api as om
import numpy as np

class HypoidPinion(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('place_coord_increm', types=int)
        self.options.declare('place_sign', types=int)
        self.options.declare('N_pinion', types=int)
        self.options.declare('l_pinion', types=float)
        self.options.declare('l_hypoid', types=float)
        self.options.declare('md_pinion', types=float)
        self.options.declare('N_hypoid', types=int)
        self.options.declare('md_hypoid', types=float)
        self.options.declare('offset', types=float)
        self.options.declare('d_pinion', types=float)
        self.options.declare('d_hypoid', types=float)

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

        N_pinion = self.options['N_pinion']
        N_hypoid = self.options['N_hypoid']
        l_pinion = self.options['l_pinion']
        l_hypoid = self.options['l_hypoid']
        md_pinion = self.options['md_pinion']
        md_hypoid = self.options['md_hypoid']
        offset = self.options['offset']
        place_coord_increm = self.options['place_coord_increm']
        place_sign = self.options['place_sign']
        d_pinion = self.options['d_pinion']
        d_hypoid = self.options['d_hypoid']

        pos_in = inputs['pos_in']
        flow_dir_in = inputs['flow_dir_in']
        motion_vector_in = inputs['motion_vector_in']
        q_dot_in = inputs['q_dot_in']

        place_dir = place_sign * np.roll(np.abs(flow_dir_in), place_coord_increm)
        flow_dir = np.cross(place_dir, flow_dir_in)

        outputs['pos_out'] = pos_in + md_hypoid * flow_dir_in + (md_pinion - l_pinion/2) * flow_dir + offset * place_dir
        outputs['flow_dir_out'] = flow_dir
        outputs['motion_vector_out'] = -1 * np.cross(place_dir, motion_vector_in)
        outputs['q_dot_out'] = (q_dot_in * N_hypoid) / N_pinion

        hypoid_axis = np.abs(motion_vector_in)
        hypoid_orthos = np.ones(3) - hypoid_axis
        bb_in_min = pos_in - 0.5 * d_hypoid * hypoid_orthos
        bb_in_max = pos_in + 0.5 * d_hypoid * hypoid_orthos

        pinion_axis = np.abs(outputs['motion_vector_out'])
        pinion_orthos = np.ones(3) - pinion_axis
        bb_out_min = outputs['pos_out'] - 0.5 * d_pinion * pinion_orthos - 0.5 * l_pinion * pinion_axis
        bb_out_max = outputs['pos_out'] + 0.5 * d_pinion * pinion_orthos + 0.5 * l_pinion * pinion_axis

        outputs['bb_min'] = np.min([bb_in_min, bb_out_min], axis=0)
        outputs['bb_max'] = np.max([bb_in_max, bb_out_max], axis=0)