from .gear_system import GearSystem
from .util.parse_input import get_num_state_var
from .util.compute_obj import compute_weight, compute_price
import openmdao.api as om
import numpy as np

class Simulator:

    def run(self, input_data):

        num_state_var = get_num_state_var(input_data)

        g = GearSystem(input_data = input_data)
        prob = om.Problem(g, reports=None)
        prob.setup()

        prob.set_val('pos_0', [0, 0, 0])
        prob.set_val('flow_dir_0', [1, 0, 0])
        prob.set_val('motion_vector_0', [1, 0, 0])
        prob.set_val('q_dot_0', 1.0)

        prob.run_model()

        output_position = np.round(prob.get_val('pos_' + str(num_state_var - 1)), 9)
        output_rot_axis = np.round(prob.get_val('motion_vector_' + str(num_state_var - 1)), 9)
        output_speed = np.round(prob.get_val('q_dot_' + str(num_state_var - 1))[0], 9)

        spatial_info = []
        bb_min_all = []
        bb_max_all = []
        for i in range(0, num_state_var-1):
            bb_min_all.append(prob.get_val('bb_min_' + str(i + 1)))
            bb_max_all.append(prob.get_val('bb_max_' + str(i + 1)))
            spatial_info.append({
                "components": g.subsys_track[i],
                "translation": prob.get_val('pos_' + str(i+1)),
                "orientation": prob.get_val('flow_dir_' + str(i+1))
            })
        bb_min = np.min(bb_min_all, axis=0)
        bb_max = np.max(bb_max_all, axis=0)

        weight = np.round(compute_weight(input_data["gear_train_sequence"]), 9)

        price = np.round(compute_price(input_data["gear_train_sequence"]), 9)

        # except:
        #     output_position = [0, 0, 0]
        #     output_rot_axis = [0, 0, 0]
        #     output_speed = 0.0
        #     weight = 0.0

        if "MRGF" in input_data["gear_train_sequence"][1]:
            input_motion_type = "T"
        else:
            input_motion_type = "R"

        if "MRGF" in input_data["gear_train_sequence"][-2]:
            output_motion_type = "T"
        else:
            output_motion_type = "R"

        return {
            "id": input_data["id"],
            "gear_train_sequence" : input_data["gear_train_sequence"],
            "input_motion_type": input_motion_type,
            "output_motion_type": output_motion_type,
            "output_position": list(output_position),
            "output_motion_vector": list(output_rot_axis),
            "output_motion_speed": output_speed,
            "weight": weight,
            "price": price,
            "bounding_box_min": list(bb_min),
            "bounding_box_max": list(bb_max)
        }
        # return spatial_info
