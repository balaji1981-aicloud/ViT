from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
app = Flask(__name__)
@app.route('/logistics', methods=["POST"])
def loading_pattern():
    try:                
        data=request.get_json()
        df1=pd.DataFrame(data)
        df=df1.copy()
        df.loc[:,'Sheets/pile'] = df['Packtype code'].astype(str).str[1:] # For march 

        df.loc[:,'Sheets/pile'] = df['Sheets/pile'].astype(float).astype(int)
        df.loc[:,'No of Piles'] = df['No.of sheets'] / df['Sheets/pile'].astype(int)
        df.loc[:,'No of Piles'] = df['No of Piles'].apply(lambda x: int(round(x)))
        def categorize_packing_type(row):
            if row['Packaging Code'] == 27 and row['Sub-Family'] != 203:
                return 'Naked'
            elif row['Packaging Code'] in [22, 18] and row['Sub-Family'] != 203:
                return 'Packed'
            elif row['Packaging Code'] in [27, 22, 18] and row['Sub-Family'] == 203:
                return 'Naked_Mirror'

        df['packing_type'] = df.apply(categorize_packing_type, axis=1)
        columns_to_read = ['Order No',  'Material Description', 'No.of sheets', 'No of Piles', 'Sheets/pile', 'Comm.Thick', 'Pack', 'Length', 'Width', 'Plant', 'packing_type', 'Sub-Family','Commercial Ton']
        result_list = []

        for value in df['Order No'].unique():

            group = df[df['Order No'] == value]
            order_dict = {
                col: group[col].tolist() for col in group.columns if col in columns_to_read
            }
            result_list.append(order_dict)
        # master_piles_df = pd.DataFrame()
        # master_positions_df = pd.DataFrame()
        # filtered_result_list = []

        # # Iterate over the original list
        # for item in result_list:
        #     packing_type = item['packing_type']
        #     # if 'Naked'or 'Naked_Mirror' in packing_type:
        #     if 'Naked' in packing_type or 'Naked_Mirror' in packing_type:
        #         filtered_result_list.append(item)

        # result_list = filtered_result_list
        print(result_list)
        master_piles_df = pd.DataFrame()
        master_positions_df = pd.DataFrame()

        for od_id in range(len(result_list)):  # len(result_list)):
            # try:    
            print('\n')
            print(f"{od_id}, Order ID: {result_list[od_id]['Order No'][0]}")
            # positions = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18]]
            description = result_list[od_id]['Material Description']
            positions = [[i] for i in range(1, int(sum(result_list[od_id]['No of Piles'])) + 1)]
            pile_values = result_list[od_id]['No of Piles']
            length_values = result_list[od_id]['Length']
            width_values = result_list[od_id]['Width']
            thickness_values = np.array(result_list[od_id]['Comm.Thick']) * np.array(result_list[od_id]['Sheets/pile'])*100
            tonnage_values = np.array(result_list[od_id]['Commercial Ton']) / np.array(result_list[od_id]['No of Piles'])
            print(tonnage_values)
            tonnage_values = np.round(tonnage_values, 2)
            packing_type = result_list[od_id]['packing_type']
            # truck_positions = {'tl': [], 'tr': [], 'ml': [], 'mr': [], 'bl': [], 'br': []}
            truck_positions = {'tl': {}, 'tr': {}, 'ml': {}, 'mr': {}, 'bl': {}, 'br': {}}


            result = {pos[0]: {'thickness': 0, 'length': 0, 'width': 0, 'tonnage': 0, 'packing_type': packing_type, 'Description':description } for pos in positions.copy()}
            for pile, thickness, length, width, tonnage, pack, desc in sorted(zip(pile_values, thickness_values, length_values, width_values, tonnage_values, packing_type, description), key=lambda x: (x[1], x[2], x[3]), reverse=True):
                for _ in range(int(pile)):
                    position = positions.pop(0)
                    if pack == 'Packed':
                        result[position[0]] = {'thickness': (thickness/100)+40, 'length': length, 'width': width, 'tonnage': tonnage, 'packing_type': pack, 'Description':desc}
                    else:
                        result[position[0]] = {'thickness': (thickness / 100), 'length': length, 'width': width, 'tonnage': tonnage, 'packing_type': pack, 'Description':desc}
            result = {pos: values for pos, values in result.items() if values['thickness'] > 0}
            print('Piles')
            for key, value in result.items():
                print(f"{key} : {value}")
            piles_df = pd.DataFrame(result.values())
            piles_df['piles'] = result.keys()
            piles_df['order_id'] = result_list[od_id]['Order No'][0]

            # master_piles_df = pd.concat([master_piles_df, piles_df], ignore_index=True)


            remaining_positions = list(result.keys())
            result_keys = list(result.keys())
            sets_list = [(entry['length'], entry['width']) for entry in result.values() if entry['length'] > 0 and entry['width'] > 0]
            set_counts = {}
            for unique_set in sets_list:
                set_counts[unique_set] = set_counts.get(unique_set, 0) + 1
            print('\nPiles and count:\n',set_counts)
            threshold = 675
            tonnage_threshold = 6.4
            paired_positions = []
            current_pair = []
            sum_list = []
            removed_values = []

            if len(remaining_positions) <13:

                count = 0
                pair_keys = []
                mirror_keys = [key for key, values in result.items() if values['packing_type'] == 'Naked_Mirror']
                pairs = {}
                i =0
                pairs_keys = []
                if mirror_keys:
                    while i<len(mirror_keys):
                        current_pair = []
                        removed_values = []
                        combined_thickness = 0
                        combined_tonnage = 0
                        # while combined_thickness<650 and i<len(mirror_keys) or i==4:
                        while combined_thickness < 650 and i < len(mirror_keys) and len(current_pair) <5 and combined_tonnage < tonnage_threshold:
                            if (combined_thickness + result[mirror_keys[i]]['thickness'] > 650) or (combined_tonnage + result[mirror_keys[i]]['tonnage'] > tonnage_threshold) or (len(current_pair) > 5):
                                break
                            combined_thickness += result[mirror_keys[i]]['thickness']
                            combined_tonnage += result[mirror_keys[i]]['tonnage']
                            current_pair.append(mirror_keys[i])
                            i = i + 1
                        max_l = 0
                        max_w = 0
                        for j in current_pair:
                            temp_l = result[j]['length']
                            temp_w = result[j]['width']
                            if temp_l > max_l:
                                max_l = temp_l
                            if temp_w > max_w:
                                max_w = temp_w
                        pair =tuple(current_pair) 
                        pairs_keys.append(pair)
                        pairs[pair] = [combined_thickness,max_l,max_w,combined_tonnage]

                if mirror_keys:
                    check = []
                    for o in pairs_keys:
                        for p in o:
                            check.append(p)
                    
                    set1 = set(check)
                    set2 = set(remaining_positions)
                    set3 = set(removed_values)
                    remaining_positions = list(set2 - set1)
                    removed_values = list(set3 - set1)

                count = 0
                pair_keys = []
                packed_keys = [key for key, values in result.items() if values['packing_type'] == 'Packed']
                # pairs = {}
                i =0
                pairs_keys = []
                if packed_keys:
                    while i<len(packed_keys):
                        current_pair = []
                        removed_values = []
                        combined_thickness = 0
                        combined_tonnage = 0
                        while combined_thickness<1050 and i<len(packed_keys) and combined_tonnage < (tonnage_threshold):
                            if (combined_thickness + result[packed_keys[i]]['thickness'] > 1050) or (combined_tonnage + result[packed_keys[i]]['tonnage'] > tonnage_threshold):
                                break
                            combined_thickness += result[packed_keys[i]]['thickness']
                            combined_tonnage += result[packed_keys[i]]['tonnage']
                            current_pair.append(packed_keys[i])
                            i = i + 1
                        max_l = 0
                        max_w = 0
                        for j in current_pair:
                            temp_l = result[j]['length']
                            temp_w = result[j]['width']
                            if temp_l > max_l:
                                max_l = temp_l
                            if temp_w > max_w:
                                max_w = temp_w
                        pair =tuple(current_pair) 
                        pairs_keys.append(pair)
                        pairs[pair] = [combined_thickness,max_l,max_w,combined_tonnage]
                
                if packed_keys:
                    check = []
                    for i in pairs_keys:
                        for j in i:
                            check.append(j)
                    
                    set1 = set(check)
                    set2 = set(remaining_positions)
                    set3 = set(removed_values)
                    remaining_positions = list(set2 - set1)
                    removed_values = list(set3 - set1)
                
                if not mirror_keys and not packed_keys:
                    pairs = {}
                    for key, value in set_counts.items():

                        if value %2 == 1:
                            current_sum = sum(sum_list) + value
                            remaining_positions.remove(sum(sum_list) + value)
                            sum_list.append(value)
                        else:
                            sum_list.append(value)

                    removed_values =  [x for x in list(result.keys()) if x not in remaining_positions]

                if len(list(result.keys())) == 10:
                    if (2440,1830) in list(set_counts.keys()) and (set_counts[(2440,1830)] > 5):
                        count = 0
                        pairs_keys = []
                        for i in list(set_counts.keys()):
                            if i == (2440,1830):
                                remaining_positions_temp = remaining_positions[count:count+set_counts[i]]
                                for j in range(0, set_counts[i], 3):
                                    if len(remaining_positions_temp[j:j+3]) == 3:
                                        if ((result[remaining_positions_temp[j]]['thickness'] + result[remaining_positions_temp[j+1]]['thickness']+result[remaining_positions_temp[j+2]]['thickness']) < 675) and ((result[remaining_positions_temp[j]]['tonnage'] + result[remaining_positions_temp[j+1]]['tonnage']+result[remaining_positions_temp[j+2]]['tonnage']) < tonnage_threshold) :
                                            pair = tuple(remaining_positions[j:j+3])
                                            pairs_keys.append(pair)
                                            pairs[pair] = [result[remaining_positions_temp[j]]['thickness'] + result[remaining_positions_temp[j+1]]['thickness']+result[remaining_positions_temp[j+2]]['thickness'],result[remaining_positions_temp[j]]['length'],result[remaining_positions_temp[j]]['width'],result[remaining_positions_temp[j]]['tonnage'] + result[remaining_positions_temp[j+1]]['tonnage']+result[remaining_positions_temp[j+2]]['tonnage']]

                                    elif len(remaining_positions_temp[j:j+3]) == 2:
                                        for l in range(len(remaining_positions_temp[j:j+2])):
                                            removed_values.append(remaining_positions_temp[j+l])

                                    elif len(remaining_positions_temp[j:j+3]) == 1:
                                        for l in range(len(remaining_positions_temp[j:j+1])):
                                            removed_values.append(remaining_positions_temp[j+l])

                            count = count + set_counts[i]
                        for i in pairs_keys:
                            for j in i:
                                remaining_positions.remove(j)
                        if removed_values:
                            for m in removed_values:
                                if m in remaining_positions:
                                    remaining_positions.remove(m)
                
                pairs_keys = []
                remaining_positions_temp = remaining_positions
                for i in range(0, len(remaining_positions_temp), 2):
                    if  (
                        len(remaining_positions_temp[i:i+2]) == 2 and 
                        (
                            result[remaining_positions_temp[i]]['length'] == result[remaining_positions_temp[i+1]]['length'] or 
                            (result[remaining_positions_temp[i]]['length'] - result[remaining_positions_temp[i+1]]['length']) < 916
                        ) and
                        (result[remaining_positions_temp[i]]['thickness'] + result[remaining_positions_temp[i+1]]['thickness']) < 700) and ((result[remaining_positions_temp[i]]['tonnage'] + result[remaining_positions_temp[i+1]]['tonnage']) < tonnage_threshold):
                        pair= (remaining_positions_temp[i], remaining_positions_temp[i + 1])
                        pairs_keys.append(pair)
                        pairs[pair] = [result[remaining_positions_temp[i]]['thickness'] + result[remaining_positions_temp[i+1]]['thickness'],result[remaining_positions_temp[i]]['length'],result[remaining_positions_temp[i]]['width'],result[remaining_positions_temp[i]]['tonnage'] + result[remaining_positions_temp[i+1]]['tonnage']]
                    
                    else:
                        for l in range(len(remaining_positions_temp[i:i+2])):
                            removed_values.append(remaining_positions_temp[i+l])
                            # remaining_positions.remove(remaining_positions_temp[i+l])
                
                for i in pairs_keys:
                    for j in i:
                        remaining_positions.remove(j)
                
                if removed_values:
                    for m in removed_values:
                        if m in remaining_positions:
                            remaining_positions.remove(m)

                print(pairs)
                if len(removed_values) > 1:
                    if removed_values:
                        removed_values_temp = removed_values.copy()
                        for i in removed_values:
                            for j in removed_values:
                                if (i != j) and (i<j) and (i in removed_values_temp) and (j in removed_values_temp):
                                    if (
                                        result[i]['length'] == result[j]['length'] and
                                        # result[i]['width'] == result[j]['width'] and
                                        (result[i]['packing_type'] == result[j]['packing_type'])and
                                        (result[i]['thickness'] + result[j]['thickness'] < 700) and
                                        (result[i]['tonnage'] + result[j]['tonnage'] < tonnage_threshold)
                                    ):
                                        pair = (i,j)
                                        pairs_keys.append(pair)

                                        pairs[pair] = [result[i]['thickness'] + result[j]['thickness'],result[j]['length'],result[j]['width'],result[i]['tonnage'] + result[j]['tonnage']]
                                        removed_values_temp.remove(i)
                                        removed_values_temp.remove(j)
                                        break

                    removed_values = removed_values_temp.copy()

                if len(removed_values) > 1:
                    if removed_values:
                        removed_values_temp = removed_values.copy()
                        for i in removed_values:
                            for j in removed_values:
                                if (i != j) and (i<j) and (i in removed_values_temp) and (j in removed_values_temp):
                                    if (
                                        (result[i]['length'] == result[j]['length'] or
                                        (result[i]['length'] - result[j]['length'])<916) and
                                        (result[i]['packing_type'] == result[j]['packing_type']) and
                                        (result[i]['thickness'] + result[j]['thickness'] < 700) and
                                        (result[i]['tonnage'] + result[j]['tonnage'] < tonnage_threshold)
                                    ):
                                        pair = (i,j)
                                        pairs_keys.append(pair)

                                        pairs[pair] = [result[i]['thickness'] + result[j]['thickness'],result[j]['length'],result[j]['width'],result[i]['tonnage'] + result[j]['tonnage']]
                                        removed_values_temp.remove(i)
                                        removed_values_temp.remove(j)
                                        break

                    removed_values = removed_values_temp.copy()

                if removed_values:
                    pairs_keys = list(pairs.keys())
                    pairs_values = list(pairs.values())
                    for i in removed_values:
                        for p in range(len(pairs_keys)):
                            if (
                                    (result[i]['length'] == pairs_values[p][1] or
                                    (result[i]['length'] - pairs_values[p][1])<916) and
                                    (result[i]['packing_type'] == result[pairs_keys[p][0]]['packing_type']) and
                                    (result[i]['thickness'] + pairs_values[p][0] < 675) and
                                    (result[i]['tonnage'] + pairs_values[p][3] < tonnage_threshold) and (len(pairs_keys[p])<3)
                                ):
                                    pairs[pairs_keys[p]] = [result[i]['thickness'] + pairs_values[p][0], max(result[i]['length'],pairs_values[p][1]) , max(result[i]['width'],pairs_values[p][2]),result[i]['tonnage'] + pairs_values[p][3]]
                                    old_key = pairs_keys[p]
                                    new_key = tuple(sorted(pairs_keys[p] + (i,)))
                                    pairs[new_key] = pairs.pop(old_key)
                                    pairs_keys.remove(pairs_keys[p])
                                    removed_values.remove(i)
                                    break

                if removed_values:
                    pairs_keys = list(pairs.keys())
                    pairs_values = list(pairs.values())
                    for i in removed_values:
                        for p in range(len(pairs_keys)):
                            if (result[i]['length'],result[i]['width']) == (result[pairs_keys[p][0]]['length'],result[pairs_keys[p][0]]['width']) and len(pairs_keys[p])<3 and (result[i]['packing_type'] == result[pairs_keys[p][0]]['packing_type']):
                                if (result[i]['thickness'] + pairs_values[p][0] < threshold) and (result[i]['tonnage'] + pairs_values[p][3] < tonnage_threshold):
                                    pairs[pairs_keys[p]] = [result[i]['thickness'] + pairs_values[p][0], result[i]['length'], result[i]['width'],result[i]['tonnage'] + pairs_values[p][3]]
                                    old_key = pairs_keys[p]
                                    new_key = tuple(sorted(pairs_keys[p] + (i,)))
                                    pairs[new_key] = pairs.pop(old_key)
                                    pairs_keys.remove(pairs_keys[p])
                                    removed_values.remove(i)
                                    break
                if removed_values:
                    for i  in removed_values:
                        pairs[(i,)] = [result[i]['thickness'],result[i]['length'],result[i]['width'],result[i]['tonnage']]

                if len(pairs.keys()) > 6 and all(entry['packing_type'] == 'Naked' for entry in result.values()):
                    pairs = {}
                    remaining_positions = list(result.keys())

                    while len(remaining_positions) >= 2:
                        pair = []
                        total_thickness = 0
                        total_tonnage = 0
                        current_max_length = 0
                        current_max_width = 0
                        for i in remaining_positions:
                            if len(pair) < 3 and (total_thickness + result[i]['thickness'] <= 700) and (total_tonnage + result[i]['tonnage'] < tonnage_threshold):
                                pair.append(i)
                                current_max_length = max(current_max_length, result[i]['length'])
                                current_max_width = max(current_max_width, result[i]['width'])
                                total_thickness += result[i]['thickness'] 
                                total_tonnage += result[i]['tonnage']
                        
                        pairs[tuple(pair)] = [total_thickness,current_max_length,current_max_width,total_tonnage]
                        current_max_length = 0
                        current_max_width = 0
                        remaining_positions = [i for i in remaining_positions if i not in pair]

                    if remaining_positions:
                        pairs[(remaining_positions[0],)] = [result[remaining_positions[0]]['thickness'],result[remaining_positions[0]]['length'],result[remaining_positions[0]]['width'],result[remaining_positions[0]]['tonnage']]

                        # pairs[remaining_positions[0]] = pairs[remaining_positions[0]] + (remaining_positions[0],)

            if len(remaining_positions) > 12:
                # for key, value in set_counts.items():

                #     if value %2 == 1:
                #         current_sum = sum(sum_list) + value
                #         remaining_positions.remove(sum(sum_list) + value)
                #         sum_list.append(value)
                #     else:
                #         sum_list.append(value)

                # removed_values =  [x for x in list(result.keys()) if x not in remaining_positions]

                count = 0
                pair_keys = []
                mirror_keys = [key for key, values in result.items() if values['packing_type'] == 'Naked_Mirror']
                packed_keys = [key for key, values in result.items() if values['packing_type'] == 'Packed']

                pairs = {}
                i =0
                pairs_keys = []
                if mirror_keys:
                    while i<len(mirror_keys):
                        current_pair = []
                        removed_values = []
                        combined_thickness = 0
                        combined_tonnage = 0
                        # while combined_thickness<650 and i<len(mirror_keys) or i==4:
                        while (combined_thickness < 650) and (i < len(mirror_keys)) and (len(current_pair) <5) and (combined_tonnage < tonnage_threshold) :
                            if (combined_thickness + result[mirror_keys[i]]['thickness'] > 650) or (combined_tonnage + result[mirror_keys[i]]['tonnage']> tonnage_threshold) or (len(current_pair) > 5):
                                break
                            combined_thickness += result[mirror_keys[i]]['thickness']
                            combined_tonnage += result[mirror_keys[i]]['tonnage']

                            current_pair.append(mirror_keys[i])
                            i = i + 1
                        max_l = 0
                        max_w = 0
                        for j in current_pair:
                            temp_l = result[j]['length']
                            temp_w = result[j]['width']
                            if temp_l > max_l:
                                max_l = temp_l
                            if temp_w > max_w:
                                max_w = temp_w
                        pair =tuple(current_pair) 
                        pairs_keys.append(pair)
                        pairs[pair] = [combined_thickness,max_l,max_w,combined_tonnage]

                if mirror_keys:
                    check = []
                    for o in pairs_keys:
                        for p in o:
                            check.append(p)
                    set1 = set(check)
                    set2 = set(remaining_positions)
                    set3 = set(removed_values)
                    remaining_positions = list(set2 - set1)
                    removed_values = list(set3 - set1)

                count = 0
                pair_keys = []
                # pairs = {}

                i =0
                pairs_keys = []
                if packed_keys:
                    while i<len(packed_keys):
                        current_pair = []
                        removed_values = []
                        combined_thickness = 0
                        combined_tonnage = 0
                        # print('before while', combined_thickness)
                        while (combined_thickness<1050) and i<len(packed_keys) and (combined_tonnage < tonnage_threshold) :
                            # print('after while', combined_thickness + result[packed_keys[i]]['thickness'])
                            if (combined_thickness + result[packed_keys[i]]['thickness'] > 1050) or (combined_tonnage + result[packed_keys[i]]['tonnage'] > tonnage_threshold) :
                                break
                            combined_thickness += result[packed_keys[i]]['thickness']
                            combined_tonnage += result[packed_keys[i]]['tonnage']
                            current_pair.append(packed_keys[i])
                            i = i + 1
                        max_l = 0
                        max_w = 0
                        for j in current_pair:
                            temp_l = result[j]['length']
                            temp_w = result[j]['width']
                            if temp_l > max_l:
                                max_l = temp_l
                            if temp_w > max_w:
                                max_w = temp_w
                        pair =tuple(current_pair) 
                        pairs_keys.append(pair)
                        pairs[pair] = [combined_thickness,max_l,max_w,combined_tonnage]
                
                if packed_keys:
                    check = []
                    for i in pairs_keys:
                        for j in i:
                            check.append(j)
                    
                    set1 = set(check)
                    set2 = set(remaining_positions)
                    set3 = set(removed_values)
                    remaining_positions = list(set2 - set1)
                    removed_values = list(set3 - set1)

                pairs_keys = []
                pair_keys = []
                for i in list(set_counts.keys()):
                    if set_counts[i]>2: #set_counts[i]%2 == 0 and 
                        remaining_positions_temp = remaining_positions[count:count+set_counts[i]]
                        # print(remaining_positions_temp, count, count+set_counts[i])
                        for j in range(0, set_counts[i], 3):
                            if len(remaining_positions_temp[j:j+3]) == 3:
                                if ((result[remaining_positions_temp[j]]['thickness'] + result[remaining_positions_temp[j+1]]['thickness']+result[remaining_positions_temp[j+2]]['thickness']) < threshold) and ((result[remaining_positions_temp[j]]['tonnage'] + result[remaining_positions_temp[j+1]]['tonnage']+result[remaining_positions_temp[j+2]]['tonnage']) < tonnage_threshold):
                                    pair = tuple(remaining_positions_temp[j:j+3])
                                    pairs_keys.append(pair)
                                    pairs[pair] = [result[remaining_positions_temp[j]]['thickness'] + result[remaining_positions_temp[j+1]]['thickness']+result[remaining_positions_temp[j+2]]['thickness'],result[remaining_positions_temp[j]]['length'],result[remaining_positions_temp[j]]['width'],result[remaining_positions_temp[j]]['tonnage'] + result[remaining_positions_temp[j+1]]['tonnage']+result[remaining_positions_temp[j+2]]['tonnage']]

                            elif len(remaining_positions_temp[j:j+3]) == 2:
                                for l in range(len(remaining_positions_temp[j:j+2])):
                                    removed_values.append(remaining_positions_temp[j+l])
                                    # remaining_positions.remove(remaining_positions_temp[j+l])
                            
                            elif len(remaining_positions_temp[j:j+3]) == 1:
                                for l in range(len(remaining_positions_temp[j:j+1])):
                                    removed_values.append(remaining_positions_temp[j+l])
                                    # remaining_positions.remove(remaining_positions[j+l])
                    count = count + set_counts[i]
                for m in pairs_keys:
                    for j in m:
                        remaining_positions.remove(j)
                if removed_values:
                    for m in removed_values:
                        if m in remaining_positions:
                            remaining_positions.remove(m)
                            
                if remaining_positions:
                    remaining_positions_temp = remaining_positions.copy()
                    for i in remaining_positions:
                        for j in remaining_positions:
                            if (i != j) and (i<j) and (i in remaining_positions_temp) and (j in remaining_positions_temp):
                                if (
                                    (result[i]['length'] == result[j]['length']) and
                                    (result[i]['width'] == result[j]['width']) and
                                    (result[i]['packing_type'] == result[j]['packing_type'])and
                                    result[i]['thickness'] + result[j]['thickness'] < 700 and result[i]['tonnage'] + result[j]['tonnage'] < tonnage_threshold
                                ):
                                    pair = (i,j)
                                    pairs_keys.append(pair)

                                    pairs[pair] = [result[i]['thickness'] + result[j]['thickness'],result[j]['length'],result[j]['width'],result[i]['tonnage'] + result[j]['tonnage']]
                                    remaining_positions_temp.remove(i)
                                    remaining_positions_temp.remove(j)
                                    break
                        if i in remaining_positions_temp:
                            removed_values.append(i)
                            
                remaining_positions = remaining_positions_temp.copy()

                if removed_values:
                    for m in removed_values:
                        if m in remaining_positions:
                            remaining_positions.remove(m)
                
                if removed_values:
                    pairs_keys = list(pairs.keys())
                    pairs_values = list(pairs.values())
                    for i in removed_values:
                        for p in range(len(pairs_keys)):
                            if (result[i]['length'],result[i]['width']) == (result[pairs_keys[p][0]]['length'],result[pairs_keys[p][0]]['width']) and len(pairs_keys[p])<3 and (result[i]['packing_type'] == result[pairs_keys[p][0]]['packing_type']):
                                if (result[i]['thickness'] + pairs_values[p][0] < threshold) and (result[i]['tonnage'] + pairs_values[p][3] < tonnage_threshold):
                                    pairs[pairs_keys[p]] = [result[i]['thickness'] + pairs_values[p][0], result[i]['length'], result[i]['width'],result[i]['tonnage'] + pairs_values[p][3]]
                                    old_key = pairs_keys[p]
                                    new_key = tuple(sorted(pairs_keys[p] + (i,)))
                                    pairs[new_key] = pairs.pop(old_key)
                                    pairs_keys.remove(pairs_keys[p])
                                    removed_values.remove(i)
                                    break

                #for 3 pairs used 611
                if removed_values:
                    pairs_keys = list(pairs.keys())
                    pairs_values = list(pairs.values())
                    for i in removed_values:
                        for p in range(len(pairs_keys)):
                            if (result[i]['length'] - result[pairs_keys[p][0]]['length'])<916 and len(pairs_keys[p])<3 and (result[i]['packing_type'] == result[pairs_keys[p][0]]['packing_type']):
                                if (result[i]['thickness'] + pairs_values[p][0] < threshold) and (result[i]['tonnage'] + pairs_values[p][3] < tonnage_threshold):
                                    pairs[pairs_keys[p]] = [result[i]['thickness'] + pairs_values[p][0], result[i]['length'], result[i]['width'],result[i]['tonnage'] + pairs_values[p][3]]
                                    old_key = pairs_keys[p]
                                    new_key = tuple(sorted(pairs_keys[p] + (i,)))
                                    pairs[new_key] = pairs.pop(old_key)
                                    pairs_keys.remove(pairs_keys[p])
                                    removed_values.remove(i)
                                    break
                                    
                if len(removed_values) > 1:
                    if removed_values:
                        removed_values_temp = removed_values.copy()
                        for i in removed_values:
                            for j in removed_values:
                                if (i != j) and (i<j) and (i in removed_values_temp) and (j in removed_values_temp):
                                    if (
                                        (result[i]['length'] == result[j]['length']) and
                                        (result[i]['width'] == result[j]['width']) and
                                        (result[i]['packing_type'] == result[j]['packing_type'])and
                                        result[i]['thickness'] + result[j]['thickness'] < 700 and
                                        result[i]['tonnage'] + result[j]['tonnage'] < tonnage_threshold
                                    ):
                                        pair = (i,j)
                                        pairs_keys.append(pair)

                                        pairs[pair] = [result[i]['thickness'] + result[j]['thickness'],result[j]['length'],result[j]['width'],result[i]['tonnage'] + result[j]['tonnage']]
                                        removed_values_temp.remove(i)
                                        removed_values_temp.remove(j)
                                        break

                    removed_values = removed_values_temp.copy()

                if len(removed_values) > 1:
                    if removed_values:
                        removed_values_temp = removed_values.copy()
                        for i in removed_values:
                            for j in removed_values:
                                if (i != j) and (i<j) and (i in removed_values_temp) and (j in removed_values_temp):
                                    if (
                                        (result[i]['length'] == result[j]['length'] or
                                        (result[i]['length'] - result[j]['length'])<916) and
                                        (result[i]['packing_type'] == result[j]['packing_type']) and
                                        result[i]['thickness'] + result[j]['thickness'] < 700 and
                                        result[i]['tonnage'] + result[j]['tonnage'] < tonnage_threshold
                                    ):
                                        pair = (i,j)
                                        pairs_keys.append(pair)

                                        pairs[pair] = [result[i]['thickness'] + result[j]['thickness'],result[j]['length'],result[j]['width'], result[i]['tonnage'] + result[j]['tonnage']]
                                        removed_values_temp.remove(i)
                                        removed_values_temp.remove(j)
                                        break

                    removed_values = removed_values_temp.copy()
                    
                if len(removed_values) == 1:
                    pairs_keys = list(pairs.keys())
                    pairs_values = list(pairs.values())
                    for i in removed_values:
                        for p in range(len(pairs_keys)):
                            if (result[i]['length'] - result[pairs_keys[p][0]]['length'])<916 and len(pairs_keys[p])<3 and (result[i]['packing_type'] == result[pairs_keys[p][0]]['packing_type']):
                                if (result[i]['thickness'] + pairs_values[p][0] < threshold) and (result[i]['tonnage'] + pairs_values[p][3] < tonnage_threshold):
                                    pairs[pairs_keys[p]] = [result[i]['thickness'] + pairs_values[p][0], result[i]['length'], result[i]['width'],result[i]['tonnage'] + pairs_values[p][3]]
                                    old_key = pairs_keys[p]
                                    new_key = tuple(sorted(pairs_keys[p] + (i,)))
                                    pairs[new_key] = pairs.pop(old_key)
                                    pairs_keys.remove(pairs_keys[p])
                                    removed_values.remove(i)
                                    break
                if removed_values:
                    for i  in removed_values:
                        pairs[(i,)] = [result[i]['thickness'],result[i]['length'],result[i]['width'],result[i]['tonnage']]

                if len(pairs.keys()) > 6 and all(entry['packing_type'] == 'Naked' for entry in result.values()):
                    pairs = {}
                    remaining_positions = list(result.keys())

                    while len(remaining_positions) >= 2:
                        pair = []
                        total_thickness = 0
                        total_tonnage = 0
                        current_max_length = 0
                        current_max_width = 0
                        for i in remaining_positions:
                            if len(pair) < 4 and (total_thickness + result[i]['thickness'] <= 675) and (total_tonnage + result[i]['tonnage'] < tonnage_threshold):
                                pair.append(i)
                                current_max_length = max(current_max_length, result[i]['length'])
                                current_max_width = max(current_max_width, result[i]['width'])
                                total_thickness += result[i]['thickness'] 
                                total_tonnage += result[i]['tonnage'] 

                        
                        pairs[tuple(pair)] = [total_thickness,current_max_length,current_max_width,total_tonnage]
                        current_max_length = 0
                        current_max_width = 0
                        remaining_positions = [i for i in remaining_positions if i not in pair]

                    if remaining_positions:
                        pairs[(remaining_positions[0],)] = [result[remaining_positions[0]]['thickness'],result[remaining_positions[0]]['length'],result[remaining_positions[0]]['width'],result[remaining_positions[0]]['tonnage']]
                        # pairs[remaining_positions[0]] = pairs[remaining_positions[0]] + (remaining_positions[0],)

                        # pairs[-1] = pairs[-1] + (remaining_positions[0],)
            # print(list(pairs.values())[3])
            pairs = dict(sorted(pairs.items(), key=lambda item: item[1][3], reverse=True))
            print('\nPairs before adding thermocol and merging PLANILAQUE:\n',pairs)

            
            # for key in pairs:
            #     # if len(key)>1:
            #     pairs[key][0] += len(key)*25
                    
            for key in pairs:
                if all(result[k]['packing_type'] != 'Packed' for k in key):
                    pairs[key][0] += len(key) * 25
                    
            pairs = dict(sorted(pairs.items(), key=lambda item: item[1][3], reverse=True))
            print('after', pairs)

            mirror_list=[]
            keys=[]
            for key in pairs.keys():
                if all(result[item]['packing_type'] == 'Naked_Mirror' for item in key) and len(key) < 6:
                    mirror_list.append(key)
            c=0
            while c < len(mirror_list):
                for k in mirror_list[c]:
                    for i in pairs.keys():
                        if all(result[val]['packing_type'] == 'Naked' for val in i if val not in keys) and (pairs[i][0] + result[k]['thickness']+25 < 750) and (pairs[i][3] + result[k]['tonnage'] < tonnage_threshold):
                            for j in pairs.keys():
                                if k in j:
                                    old_key = j
                                    pairs[j][0] = pairs[j][0] - result[k]['thickness'] -25
                                    break
                            
                            new_key = list(j)
                            new_key.remove(k)
                            new_key = tuple(new_key)

                            pairs[new_key] = pairs.pop(j)

                            pairs[tuple(sorted(i + (k,)))] = [pairs[i][0] + result[k]['thickness']+25, max(pairs[i][1], result[k]['length']), max(pairs[i][2],result[k]['width']),pairs[i][3] + result[k]['tonnage']]
                            pairs.pop(i)
                            keys.append(k)
                            break
                c+=1
            for k in list(pairs.keys()):
                if k == ():
                    del pairs[k]
            # print('!!!!!!!!!!',pairs[-1])
            pairs = dict(sorted(pairs.items(), key=lambda item: item[1][3]))
            print('\nPairs after adding thermocol and merging PLANILAQUE:\n',pairs)
            print('\nTotal No of Pairs:',len(pairs))
            pairs_df = pd.DataFrame.from_dict(pairs, orient='index', columns=['Total Thickness', 'Width', 'Height', 'Tonnage'])
            pairs_df.reset_index(inplace=True)
            pairs_df.rename(columns={'index': 'Pair'}, inplace=True)
            # split = 0
            # if len(pairs)%2!=0 and len(pairs)>1 and pairs_df.iloc[-1]["Tonnage"]>2 and len(pairs_df['Pair'].iloc[-1])>1:
            #     # split = 1
            #     print('split is done')
            #     split_pairs = pairs_df.iloc[-1]["Pair"]
            #     if len(split_pairs)%2==0:
            #         row1 = split_pairs[:-(int(len(split_pairs)/2))]
            #         row2 = split_pairs[(int(len(split_pairs)/2)):]
            #     else:
            #         row1 = split_pairs[:-(int(len(split_pairs)/2))-1]
            #         row2 = split_pairs[(int(len(split_pairs)/2)):]
            #     # print((int(len(split_pairs)/2)))
            #     for rows in [row1,row2]:
            #         row_total_thickness = 0
            #         row_width = []
            #         row_length = []
            #         row_tonnage = 0
            #         for i in range(len(rows)):
            #             row_total_thickness += result[rows[i]]['thickness'] + 25
            #             row_length.append(result[rows[i]]['length'])
            #             row_width.append(result[rows[i]]['width'])
            #             row_tonnage += result[rows[i]]['tonnage']

            #         row_max_length = max(row_length)
            #         row_max_width = max(row_width)
            #         if rows[i] in row1:
            #             pairs_df = pairs_df.drop(pairs_df.index[-1])
            #             pairs_df.loc[len(pairs_df)] = [rows, row_total_thickness, row_max_length, row_max_width, row_tonnage ]
            #         else:
            #             pairs_df.loc[len(pairs_df)] = [rows, row_total_thickness, row_max_length, row_max_width, row_tonnage]
                # print(pairs_df)

            position_keys = list(truck_positions.keys())
            position_index = 0
            for key, value in pairs.items():
                current_position = position_keys[position_index]
                truck_positions[current_position][f'pair: {key}'] = {
                    'thickness': value[0],
                    'length': value[1],
                    'width': value[2],
                    'tonnage': value[3]
                }
                position_index += 1
                position_index %= len(position_keys)
            
            truck_positions = {key: value for key, value in truck_positions.items() if value}
            # print('OLD',truck_positions)
            truck_posi = ['tr','tl', 'ml', 'mr', 'br', 'bl']

            if not pairs_df.empty and len(truck_posi)>=len(pairs_df):
                pairs_df['truck_positions'] = np.nan
                # print('Truck evened')
                pairs_df['truck_positions'] = truck_posi[:len(pairs_df)]
                # print(pairs_df)
                truck_positions = {
                    row['truck_positions']: {
                        f"pair: {row['Pair']}": {
                            'thickness': row['Total Thickness'],
                            'length': row['Width'],
                            'width': row['Height'],
                            'tonnage': row['Tonnage']
                        }
                    }
                    for _, row in pairs_df.iterrows()
                }
                # print('NEW',truck_positions)
                if not pairs_df['truck_positions'].empty: 
                    t_length_l = sum(pairs_df[pairs_df['truck_positions'].str.contains('l')]['Width'])
                    # print(t_length_l)
                    t_length_r = sum(pairs_df[pairs_df['truck_positions'].str.contains('r')]['Width'])
            else:
                # print('No NEW')
                left = {}
                right = {}

                for val in list(truck_positions.keys())[::2]:
                    left[val] = truck_positions[val]

                for val in list(truck_positions.keys())[1:6:2]:
                    right[val] = truck_positions[val]

                # t_length_l = sum(data['length'] for data in left.values())
                # t_length_r = sum(data['length'] for data in right.values())

                t_length_l = sum(data['length'] for d in left.values() for data in d.values())
                t_length_r = sum(data['length'] for d in right.values() for data in d.values())
            

            t_length = max(t_length_l,t_length_r)

            # print('t_length',t_length)
            used_area = 0
            print('\nTruck_positions:\n',truck_positions)



            # Iterate through the dictionary
            for position in truck_positions.values():
                # print('position', position)
                # print('position_keys',position.keys())
                for pair_data in position.values():
                    # print('pair_data', pair_data)
                    thickness = pair_data['thickness']
                    # print(thickness, used_area)
                    if (thickness > used_area) and (thickness < 750) :
                        used_area = thickness

            unused_area = 750 - used_area
            
            if unused_area < 251:
                supervised_needed_area = 250 - unused_area 
            elif unused_area > 251:
                supervised_needed_area = 0
            if unused_area <301:
                safe_needed_area = 300 - unused_area
            elif unused_area >301 :
                safe_needed_area = 0

            t_width_supervised = used_area + unused_area + supervised_needed_area

            if t_width_supervised < 831:
                t_width_supervised = 830*2 + 450
            else:
                t_width_supervised = t_width_supervised + t_width_supervised + 450  

            t_width_safe = used_area + unused_area + safe_needed_area
            
            if t_width_safe < 831:
                t_width_safe = 830*2 + 450
            else:
                t_width_safe = t_width_safe + t_width_safe + 450

            if len(truck_positions) <3:
                safe = 600
                supervised = 400
            elif len(truck_positions) > 2 and len(truck_positions) <5:
                safe = 1100
                supervised = 700
            elif len(truck_positions) >4:
                safe = 1600
                supervised = 1000
            if len(pairs)<7:
                print('!!!!!',len(truck_positions))

                result_list[od_id]['Safe_loading'] = [f'{t_length + safe} * {t_width_safe}']*len(result_list[od_id]['Order No'])
                result_list[od_id]['Supervised_loading'] = [f'{t_length + supervised} * {t_width_supervised}']*len(result_list[od_id]['Order No'])
                Safe_loading = f'{t_length + safe} * {t_width_safe}'
                Supervised_loading = f'{t_length + supervised} * {t_width_supervised}'

                positions_df = pd.DataFrame([
                {'position': pos, 'pair': key.replace('pair: (', '').replace(')', ''), 'Order No': result_list[od_id]['Order No'][0], **value, 'Safe_loading': Safe_loading, **value, 'Supervised_loading': Supervised_loading, **value} 
                for pos, inner_dict in truck_positions.items() 
                for key, value in inner_dict.items()
                ])

                positions_df['pair'] = positions_df['pair'].str.replace('pair: ', '').str.replace('\(', '').str.replace('\)', '')
                
                master_piles_df = pd.concat([master_piles_df, piles_df], ignore_index=True)
                master_positions_df = pd.concat([master_positions_df, positions_df], ignore_index=True)

                print(f'\nSafe Loading {t_length + safe} * {t_width_safe}')
                print(f'Supervised Loading {t_length + supervised} * {t_width_supervised}')
            final = pd.DataFrame(result_list)
            final = final.apply(lambda x: pd.Series(x).explode(ignore_index=True))
            # df['Safe_loading'] = final['Safe_loading']
            # df['Supervised_loading'] = final['Supervised_loading']
            final = final.drop_duplicates(subset = 'Order No')
            merged_df = pd.merge(df1, final[['Order No','Safe_loading','Supervised_loading']], on='Order No', how='left')
            final = final.drop_duplicates(subset = 'Order No')
            merged_df = pd.merge(df1, final[['Order No','Safe_loading','Supervised_loading']], on='Order No', how='left')

            # Combine them into a single dictionary
            output_df = {
                'raw': merged_df.to_json(orient='records'),
                'piles': master_piles_df.to_json(orient='records'),
                'positions': master_positions_df.to_json(orient='records')
            }
        return output_df       
    except:
        return "Bad request", 400
if __name__ == '__main__':
    app.run(port=5002,use_reloader=False,debug=True)