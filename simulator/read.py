import h5py

for i in range(0, 2):
    fname = "rm_data_sim_" + str(i) + ".h5"
    with h5py.File(fname, 'r') as hdf5_file:
        key_to_access = "sample_0"
        grp = hdf5_file[key_to_access]
    
        # Access each dataset
        attr1 = grp['req_input'][:]
        attr2 = grp['orig_seq'][:]
        attr3 = grp['pred_seq'][:]  
        attr4 = grp['logits'][:]  # Convert to tensor if needed
        # attr5 = grp['pred_seq_idx'][:]  
        attr6 = grp['pred_output_motion_speed'][()]
        attr7 = grp['pred_output_position'][:]
        attr8 = grp['pred_weight'][()]

        print(f"Key: {key_to_access}")
        print(f"attr1: {attr1}")
        print(f"attr2: {attr2}")
        print(f"attr3: {attr3}")
        print(f"attr4: {attr4[0]}")
        # print(f"attr5: {attr5}")
        print(f"attr6: {attr6}")
        print(f"attr7: {attr7}")
        print(f"attr8: {attr8}")
