import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ortools.linear_solver import pywraplp
import time

def on_change_checkbox():
    st.session_state['buffer'] = True

def on_change_genre():
    st.session_state['genre'] = True

# @st.cache_data
def buffer(data):
    if st.session_state['buffer']:        
        # print(df.columns)
        temp_df = pd.DataFrame(columns=['Material Code','Block Length'])
        for i in data['Material Code'].unique():
            temp = data[data['Material Code'] == i]
            temp.reset_index(inplace=True,drop=True)
            st.session_state['min_index'] = np.argmin(temp['Total wastage'])
            min_x =temp['Block_size'][st.session_state['min_index']]
            temp_df = temp_df._append(pd.DataFrame([[i,min_x]],columns=['Material Code','Block Length']))
        temp_df.to_csv("buffer_data.csv")
        st.session_state['buffer'] = False
        return temp_df
    else:
        temp_df = pd.read_csv("buffer_data.csv")
        temp_df.index=[0]*len(temp_df)
        temp_df.drop(['Unnamed: 0'],axis=1,inplace=True)
        return temp_df

@st.cache_data
def upload(uploaded_files):
    if uploaded_files:
        # print(uploaded_files)
        csv_reports = []
        excel_reports = []
        count=0
        for i in uploaded_files:
            if i.name.endswith('.csv'):
                csv_reports.append(i.name)
            elif i.name.endswith('.xlsx'):
                excel_reports.append(i.name)
        
        for i in range(len(csv_reports)):
            temp = pd.read_csv(uploaded_files[i])
            pattern = r'\d+_(.*)$'

            # Extract names after numbers
            match = re.search(pattern, csv_reports[i])

            # Check if match is found
            if match:
                names = match.group(1)
                names = re.sub(r'\d+_','', names)
            else:
                st.write("Customer Name Missing")
            temp['customer_name'] = names

            if count == 0:
                df = temp
                count = 1
            else:
                df=pd.concat([df,temp],ignore_index=True)

        for i in range(len(excel_reports)):
            temp = pd.read_excel(uploaded_files[i] )
            pattern = r'\d+_(.*)$'

            # Extract names after numbers
            match = re.search(pattern, excel_reports[i])

            # Check if match is found
            if match:
                names = match.group(1)
                names = re.sub(r'\d+_','', names)
            else:
                st.write("Customer Name Missing")
            temp['customer_name'] = names

            if count == 0:
                df = temp
                count = 1
            else:
                df=pd.concat([df,temp],ignore_index=True)

        df['customer_name'] = df['customer_name'].apply(lambda x : ' '.join(re.split(r"\.",x)[:-1]))
        return df

def create_solver(solver_type,pieces,standard_block):    
    solver = pywraplp.Solver.CreateSolver(solver_type)

    # Define variables
    blocks_used = [solver.BoolVar(f'Block_{i}') for i in range(len(pieces))]
    pieces_cut = [[solver.BoolVar(f'Piece_{i}_{j}') for j in range(len(pieces))] for i in range(len(pieces))]

    # Define objective: minimize the total number of standard blocks used and total wastage
    objective = solver.Objective()
    objective.SetMinimization()

    # Coefficients for the number of blocks used
    for block in blocks_used:
        objective.SetCoefficient(block, 1)

    # Coefficients for the total wastage
    wastage_vars = [solver.NumVar(0, solver.infinity(), f'Wastage_{i}') for i in range(len(pieces))]
    for j in range(len(pieces)):
        solver.Add(wastage_vars[j] == standard_block - solver.Sum(pieces[i] * pieces_cut[i][j] for i in range(len(pieces))))
        objective.SetCoefficient(wastage_vars[j], 1)

    # Add constraints
    for i in range(len(pieces)):
        solver.Add(solver.Sum(pieces_cut[i]) == 1)

    for j in range(len(pieces)):
        solver.Add(solver.Sum(pieces[i] * pieces_cut[i][j] for i in range(len(pieces))) <= standard_block * blocks_used[j])

    return solver,wastage_vars,pieces_cut,blocks_used

@st.cache_data       
def optimization(df):            
        st.session_state['buffer'] = True
        st.session_state['genre'] = True
        wastage_final_df=pd.DataFrame()
        final_allocation_df=pd.DataFrame()
        for val in list(df['Material Code'].unique()):            
            pieces=list(df[df['Material Code']==val]['Length(mm)'])
        #     combos=list(df[df['Material Code']==val]['combo'])
            pieces=[num + 10 for num in pieces]
            
            list_final=[]
            # Define the range of standard lengths
            standard_lengths = list(range(5000, 7001, 100))
        #     print(standard_lengths)
            # Shuffle the order of pieces
            random.shuffle(pieces)

            for i,standard_block in enumerate(standard_lengths):                             
                solver,wastage_vars,pieces_cut,blocks_used =create_solver('SAT',pieces,standard_block)
                solver.SetTimeLimit(30 * 1000)
                # Solve the problem
                start_time = time.time()  # Start time tracking
                status = solver.Solve()
                elapsed_time = time.time() - start_time
                if status != pywraplp.Solver.OPTIMAL and elapsed_time >= 30:                    
                    solver, wastage_vars, pieces_cut, blocks_used = create_solver('SCIP', pieces, standard_block)
                    solver.SetTimeLimit(30 * 1000)  # Set the time limit for SCIP as well
                    status = solver.Solve()  # Solve again with SCIP solver
                dummy_df=pd.DataFrame(columns=['Material Code','Block_size','Total wastage','num_blocks_used','Distributions'])
                if status == pywraplp.Solver.OPTIMAL:
                    num_blocks_used = sum([block.solution_value() for block in blocks_used])
                    total_wastage = sum(standard_block - wastage_vars[j].solution_value() for j in range(len(pieces)))
                    # Collect allocation information into a dataframe
                    allocation_data = {'Piece': [], 'Length(mm)': [], 'Block': []}
                    for i in range(len(pieces)):
                        for j in range(len(pieces)):
                            if pieces_cut[i][j].solution_value() > 0:
                                allocation_data['Piece'].append(i+1)
                                allocation_data['Length(mm)'].append(pieces[i])
                                allocation_data['Block'].append(j+1)
                    allocation_df = pd.DataFrame(allocation_data)
                    allocation_df['Length(mm)']=allocation_df['Length(mm)']-10
                    allocation_df['Block_Length']=standard_block
                    allocation_df['material_code']=val                             
                    grouped_dict = {group: values['Length(mm)'].tolist() for group, values in allocation_df.groupby('Block')}
        #             print(grouped_dict)
                    # Collect wastage information into a dataframe
                    wastage_data = {'Block': [], 'Usage': []}
                
                    final_allocation_df=pd.concat([allocation_df,final_allocation_df],axis=0)
                
                    for j in range(len(pieces)):
                        if blocks_used[j].solution_value() == 1:
                            block_wastage = standard_block - wastage_vars[j].solution_value()
                            wastage_data['Block'].append(j+1)
                            wastage_data['Usage'].append(block_wastage)
                    wastage_df = pd.DataFrame(wastage_data)
                    wastage_df['Wastage']=standard_block - wastage_df['Usage']
                    wastage_df.sort_values(by='Wastage',inplace=True)
                    wastage_df['Block'] = range(1, len(wastage_df) + 1)
                    list_final.append(wastage_df)
                    dummy_df.loc[i,'Total wastage']=int(((num_blocks_used*standard_block)- total_wastage))
                else:
                    print('The problem does not have an optimal solution.')
                    num_blocks_used=-1                    
                    grouped_dict={}
                    dummy_df.loc[i,'Total wastage']=-1
                # dummy_df=pd.DataFrame(columns=['Material Code','Block_size','Total wastage','num_blocks_used','Distributions'])
                dummy_df.loc[i,'Material Code']=val
                dummy_df.loc[i,'Block_size']=standard_block
                # dummy_df.loc[i,'Total wastage']=int(((num_blocks_used*standard_block)- total_wastage))
                dummy_df.loc[i,'num_blocks_used']=int(num_blocks_used)
                dummy_df.loc[i,'Distributions']=[grouped_dict]
                wastage_final_df=pd.concat([wastage_final_df,dummy_df],axis=0)

        final_df=pd.DataFrame()
        for val in list(final_allocation_df['material_code'].unique()):
            standard_lengths = list(range(5000, 7001, 100))
            for standard_block in standard_lengths:  
                    df_base=final_allocation_df[(final_allocation_df['Block_Length']==standard_block) & (final_allocation_df['material_code']==val)]
                    # print(df)
                    test = df[['Length(mm)','customer_name','Material Code']]
                    df_cust=test[test['Material Code']==val]
                    # Step 1: Find common IDs between both dataframes
                    common_ids = [id_val for id_val in df_base['Length(mm)'] if id_val in df_cust['Length(mm)'].tolist()]                
                    # Step 2: Merge rows from df_cust to df_base based on ID
                    merged_df = pd.merge(df_base, df_cust, on='Length(mm)', how='inner')
                    merged_df=merged_df.drop_duplicates()
                    merged_df.reset_index(drop=True,inplace=True)                
                    # Step 3: Remove corresponding rows from df_cust
                    df_cust_filtered = df_cust[~df_cust['Length(mm)'].isin(common_ids)]
                    final_df=pd.concat([merged_df,final_df],axis=0)
                    final_df.drop(columns=['material_code'],inplace=True)
                    final_df['Block'] = final_df.groupby(['Block_Length','Material Code'])['Block'].rank(method='dense').astype(int)
        
        return wastage_final_df, final_df

def convert_df(df):
   df.rename(columns={"Block":"Bars","Block_Length":"Bar_Length"},inplace=True)
   return df.to_csv(index=False).encode('utf-8')

# Function to apply style to rows with multiple values
def highlight_rows(x):
    df = pd.DataFrame('', index=x.index, columns=x.columns)
    for index in indexes_with_multiple_values:
        df.loc[index, :] = 'background-color: lightgreen'
    return df
    
if __name__=='__main__':

    st.set_page_config(layout="wide")    
    image1 = r".\Logos\sg.jpg"

    col1, col2,_ = st.columns(3)

    with col1:
        st.image(image1, width=200)

    with col2:
        st.markdown("<h2 style='text-align: center; color: grey;'>Windows Profile Optimization</h2>", unsafe_allow_html=True)

    uploaded_files = st.sidebar.file_uploader(":blue[Choose multiple files to upload:]", accept_multiple_files=True,)
    generate_button = st.sidebar.button("Upload")
    st.session_state["selected_option"] = ''
    if generate_button or 'key' in st.session_state:
        st.session_state['key'] = 'value'
        # st.session_state['genre'] = True

        if uploaded_files:
            data = upload(uploaded_files)
            print("columns",data.columns)
            data.rename(columns={"rmcode":"Material Code","barcodelength":"Length(mm)"},inplace=True)
            print("before",len(data))
            data=data[(data['Material Code']==58010001) & (data['workordno']=="WO-10195")]
            print("after",len(data))
            pivot =  pd.pivot_table(data, values='Length(mm)', index='Material Code', columns='customer_name',
                          aggfunc='count')
            
            pivot.fillna('',inplace=True)
            pivot = pivot.astype("str").replace(r'\.0','',regex=True)

            # Create a boolean mask where values are not null
            not_null_mask =  pivot.applymap(lambda x: pd.notna(x) and x != '')

            # Sum the boolean mask across the columns
            sum_mask = not_null_mask.sum(axis=1)

            # Identify indexes where the sum is greater than 1
            indexes_with_multiple_values = pivot.index[sum_mask > 1].tolist()

            pivot = pivot.style.apply(highlight_rows, axis=None)

            
            # Display the table
            st.table(pivot)

            # Dynamically create checkboxes based on the length of the DataFrame columns
            checkboxes = {}
            cols = st.columns(len(pivot.columns)+1)

            for idx, column in enumerate(pivot.columns):
                checkboxes[column] = cols[idx+1].checkbox(f"{column}", key=column,on_change=on_change_checkbox)

            # Filter dataframe based on selected checkboxes
            selected_columns = [column for column, checked in checkboxes.items() if checked]
            # Create a horizontal divider
            data =  data[data['customer_name'].apply(lambda x: x in selected_columns)]
            st.write("---")
            if len(selected_columns)>=1:
                wastage_final_df, final_df = optimization(data)

                min_table = buffer(wastage_final_df)

                try:
                    wastage_data = wastage_final_df.copy()
                    final_data = final_df.copy()

                    options = wastage_data['Material Code'].unique()
                    selected_option = st.selectbox(
                        ":blue[Select a Material Code:]",
                        options=options,on_change=on_change_genre,
                    )

                    if selected_option:
                        st.session_state["selected_option"] = selected_option

                    wastage_data = wastage_data[wastage_data['Material Code']==selected_option]
                    wastage_data.sort_values(['Block_size'],inplace=True)
                    wastage_data.reset_index(inplace=True,drop=True)
                    
                    # if st.session_state["selected_option"] != selected_option:
                    radio_key = {
                        5000: 0,
                        5100: 1,
                        5200: 2,
                        5300: 3,
                        5400: 4,
                        5500: 5,
                        5600: 6,
                        5700: 7,
                        5800: 8,
                        5900: 9,
                        6000: 10,
                        6100: 11,
                        6200: 12,
                        6300: 13,
                        6400: 14,
                        6500: 15,
                        6600: 16,
                        6700: 17,
                        6800: 18,
                        6900: 19,
                        7000: 20
                    }
                    if st.session_state['genre']:
                        st.session_state['min_index'] = wastage_data[wastage_data['Block_size'] == min_table['Block Length'][min_table['Material Code'] == selected_option][0]].index.values[0]
                        

                        min_x = wastage_data['Block_size'][st.session_state['min_index']]
                        min_y = wastage_data['Total wastage'][st.session_state['min_index']]
                        st.session_state['genre'] = False
                        
                    # print(st.session_state['min_index'],radio_key[wastage_data['Block_size'].get(st.session_state['min_index'])])
                    genre = st.radio(
                    ":blue[Bar Size]",
                    wastage_data['Block_size'].unique(),
                    index = radio_key[wastage_data['Block_size'].get(st.session_state['min_index'])],
                    horizontal=True)

                    
                    min_table['Block Length'][(min_table['Material Code'] == selected_option)] = genre
                    st.session_state['min_index'] = wastage_data[wastage_data['Block_size'] == genre].index.values[0]
                    min_x = wastage_data['Block_size'][st.session_state['min_index']]
                    min_y = wastage_data['Total wastage'][st.session_state['min_index']]
                    min_table.to_csv("buffer_data.csv")
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])

                    # Add traces
                    fig.add_trace(
                        go.Scatter(x=wastage_data['Block_size'], y=wastage_data['Total wastage'], name="wastage"),
                        secondary_y=False,
                    )

                    fig.add_trace(
                        go.Scatter(x=wastage_data['Block_size'], y=wastage_data['num_blocks_used'], name="blocks"),
                        secondary_y=True,
                    )

                    # Add figure title
                    fig.update_layout(
                        title_text="Total Wastage corresponding to Number of Bars Used"
                    )

                    # Set x-axis title
                    fig.update_xaxes(title_text="Bar Size")

                    # Set y-axes titles
                    fig.update_yaxes(showgrid=False, title_text="Total Wastage", secondary_y=False)
                    fig.update_yaxes(showgrid=False, title_text="Number of Bars Used", secondary_y=True)

                    # Add a separate scatter trace for the minimum value, with visibility toggled
                    min_trace = px.scatter(x=[min_x], y=[min_y]).data[0]
                    min_trace['visible'] = True  # Explicitly make the minimum point visible
                    min_trace['marker'] = marker=dict(
                            color='black',  # Set marker color for minimum value
                            symbol='star',  # Set marker symbol for minimum value
                            size=12 # Set marker size for minimum value
                        )
                    
                    fig.add_trace(min_trace)
                    fig.update_layout(legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ))

                    st.plotly_chart(fig, use_container_width=True)

                    min_table.index = np.arange(1,len(min_table)+1)
                    st.sidebar.table(min_table)
                    
                    count=0
                    for i in min_table.values:
                        if count == 0:
                            temp_df = final_data[(final_data['Material Code'] == i[0])&(final_data['Block_Length'] == i[1])]
                            count=1
                        else:
                            temp_df = temp_df._append(final_data[(final_data['Material Code'] == i[0])&(final_data['Block_Length'] == i[1])])
                    
                    csv = convert_df(temp_df)

                    st.download_button(
                    "Download Final Data",
                    csv,
                    "final_data.csv",
                    "text/csv",
                    key='download-csv'
                    )
                except FileNotFoundError:
                    pass
        else:
            pass

