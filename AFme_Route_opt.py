from flask import Flask,request,jsonify
from sklearn.neighbors import DistanceMetric
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing
import pandas as pd
import numpy as np
from datetime import date,datetime
import geopy.distance
import json
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/route_optimizer', methods=["POST"])
def route_optimization():
    try:
        data=request.get_json()
        lat=data["start_lat"]
        long=data["start_lon"]
        plant_desc=data["plant_desc"]
        type_cluster=data["type_cluster"]
        clusters=data["clusters"]
        region=data["region"]
        file=data["invoice_records"]
        try:
            if ((lat==12.934787) & (long==79.914201)):
                destination="Chennai"
            else:
                destination=None
            df=pd.DataFrame(file)
            df_data=df[['invoice_no',  'destination',
        'cluster_name', 'ship_to_code', 'sold_to_code', 
        'ship_to_name',  'plant_description', 'region',
              'pend_qty',  'customer_name', 'expected_date','order_number',
         'call_center_remarks', 'latitude','longitude','collection_scope']]
            if plant_desc!="":
                list1=[]
                if len(plant_desc)>0:
                    for i in plant_desc:
                        if i=='Sriperumbudur':
                             i='G500'
                        elif i=='Bhiwadi':
                             i='G501'
                        else:
                             i='G505'
                        list1.append(df_data[df_data['collection_scope']==i])         
                    df_data=pd.concat(list1)                    
                # else:
                #     df_data=df_data

            list_region=[]
            if len(region)>0:
                for i in region:
                    list_region.append(df_data[df_data['region']==i])         
                df_data=pd.concat(list_region)     


            #Pending quantity
            df_data_pend=df_data[df_data['pend_qty']!=0]

            #Intransit removed
            df_data_transit=df_data_pend[(df_data_pend['call_center_remarks']!='IN TRANSIT')]                 
            df_data_transit['Expected Date'] = pd.to_datetime(df_data_transit['expected_date'], format="%d.%m.%Y")            
            today = date.today().strftime("%Y-%m-%d")
            df_data_transit['Transit']=np.where(df_data_transit['Expected Date']>today,"IN TRANSIT", "TRANSIT COMPLETED")
            df_data_transit=df_data_transit[(df_data_transit['Transit']!='IN TRANSIT')]

            #Disputes removed
            df_data_dispute=df_data_transit[df_data_transit['call_center_remarks']!='DISPUTE FRAMES - UNKNOWN']

            #Dropped at drop center
            df_data_dropcenter=df_data_dispute[df_data_dispute['call_center_remarks']!='DROPPED @ DROP CENTER']

            #A Frames used by customers removed
            df_data_used=df_data_dropcenter[(df_data_dropcenter['call_center_remarks']!='USED BY CUSTOMER')]

            df_data_used=df_data_used[df_data_used['order_number']==""]

            df_data_used=df_data_used[['cluster_name','latitude','longitude','destination','ship_to_name','pend_qty','region']]
            df_data_used.dropna(inplace=True)
            df_data_used['pend_qty']=df_data_used['pend_qty'].astype(int)                        
            def test(loc_df,point1,df_data_used,dist,clusters,lat,long):      
                loc_df=loc_df[~loc_df['lat_lon'].isin([start_point])]        
                loc_df.reset_index(drop=True,inplace=True)
                df1 = pd.DataFrame([{"lat_lon":start_point}],index =[0])
                loc_df=pd.concat([df1,loc_df]).reset_index(drop = True)   
                points=loc_df['lat_lon'].to_list()
                points.remove(start_point)
                points.insert(0,start_point)
                points.append(end_point)

                dummy_node = (30.353603, 90.108266)
                points.append(dummy_node)

                distance_matrix = np.zeros((len(points), len(points)))
                for i in range(len(points)):
                    for j in range(len(points)):
                        lat1, lon1 = points[i]
                        lat2, lon2 = points[j]
                        distance_matrix[i][j] = distance_coords((lat1, lon1), (lat2, lon2))
                for i in range(len(points)):
                    distance_matrix[i][points.index(start_point)] = 0
                    distance_matrix[points.index(end_point)][i] = 0
                
                if len(loc_df)<20:                
                    solution,out = solve_tsp_dynamic_programming(distance_matrix)
                    loc_df['NewIndex'] = None
                    for indx, value in enumerate(solution):
                        loc_df['NewIndex'][value] = indx
                        loc_df = loc_df.sort_values(by=['NewIndex'])
                    #         loc_df = loc_df.tail(-1)
                    df_arr_seq=loc_df.merge(df_data_used,on=['lat_lon'],how='left')       
                    df_arr_seq.drop(['NewIndex','Latitude_rad','Longitude_rad'], axis=1,inplace=True) 
                    df_arr_seq.dropna(inplace=True)                    
                else :            
                    dist1=[]
                    per=[]
                    for i in range(5):            
                        permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
                        dist1.append(distance)
                        per.append(permutation)
                    permutation=per[dist1.index(min(dist1))]

                    loc_df['NewIndex'] = None
                    for indx, value in enumerate(permutation):
                        loc_df['NewIndex'][value] = indx
                        loc_df = loc_df.sort_values(by=['NewIndex'])            
                    df_arr_seq=loc_df.merge(df_data_used,on=['lat_lon'],how='left')   
                    df_arr_seq.drop(['NewIndex','Latitude_rad','Longitude_rad'], axis=1,inplace=True)        

                sums = 0        
                old_index = 0
                df_list=[]
                for index, i in enumerate(df_arr_seq['pend_qty']): 
                    sums += int(i)        
                    if sums >= 15:
                        df_list.append(df_arr_seq[old_index:index+1])
                        old_index = index + 1
                        sums = 0
                df_list.append(df_arr_seq[old_index:])
                data = [df1 for df1 in df_list if not df1.empty]

                if len(data)>0: 
                    for index,value in enumerate(data):
                        value['truck']="Truck " + str(index+1)
                        df_append=pd.DataFrame([{"lat_lon":end_point,"latitude":end_point[0],"longitude":end_point[1],"truck" : "Truck " + str(index+1),"region":region[0],"destination":destination}])
                        data[index] = pd.concat([value, df_append], ignore_index=True)        

                    df_fin=pd.concat(data)                    
                    groups=df_fin.groupby("truck")
                    MOQ=groups.filter(lambda g: g['pend_qty'].sum() <15.)
                    df_fin=pd.concat([df_fin,MOQ]).drop_duplicates(keep=False)

                    MOQ['truck']=" ".join(clusters).replace(" cluster","").replace(" Cluster","") + " cluster MOQ"

                    df_fin=pd.concat([df_fin,MOQ])
                    df_fin['sequence']=df_fin.groupby('truck').cumcount()+1                     
                return df_fin
            
            def distance_coords(coords_1,coords_2):
                return geopy.distance.geodesic(coords_1, coords_2).km

            if ((lat is None)  or (long is None)):
                point1=(12.934787,79.914201)
            else:    
                 point1=(lat,long)
            end_point=point1            
            df_list2=[]
            df_list=[]
            df_data_used.dropna(inplace=True)
            df_data_used['latitude']=df_data_used['latitude'].astype(float)
            df_data_used['longitude']=df_data_used['longitude'].astype(float)
            
            df_data_used=df_data_used.groupby(['cluster_name','region','destination', 'ship_to_name',  'latitude', 'longitude']).sum().reset_index()

            df_data_used['lat_lon'] = list(zip(df_data_used.latitude, df_data_used.longitude))
            df_data_used['Latitude_rad'] = np.radians(df_data_used['latitude'])
            df_data_used['Longitude_rad'] = np.radians(df_data_used['longitude'])
            dist = DistanceMetric.get_metric('haversine')            
            out=[distance_coords(point1,value) for value in df_data_used['lat_lon'].to_list()]
            max_index = out.index(max(out))
            point1=df_data_used.loc[max_index]['lat_lon']
            start_point=point1            
            if clusters is not None:
                if len(clusters)>0:
                        if type_cluster=='inter':
                            df_list2=[]
                            for i in clusters:                                                     
                                loc_df = df_data_used[df_data_used['cluster_name']==i][['lat_lon']]                   
                                df_list2.append(loc_df)
                            loc_df=pd.concat(df_list2)
                            df_fin=test(loc_df,point1,df_data_used,dist,clusters,lat,long)
                        else:                
                            for i in clusters:                                  
                                loc_df = df_data_used[df_data_used['cluster_name']==i][['lat_lon']]   
                                if len(loc_df)>0:
                                    df_fin=test(loc_df,point1,df_data_used,dist,clusters,lat,long)            
                else :
                    df_fin=pd.DataFrame({"Latitude":[lat],"Longitude":[long]})
            
            elif clusters is None:
                df_fin=pd.DataFrame({"Latitude":[15.001200],"Longitude":[80.256500]})            
            
            if len(df_fin)>0:
                df_fin.drop(columns=["lat_lon"],inplace=True)
                json=df_fin.to_json(orient='records')
                return json
            else:
                json="No data found"
            return json, 200
        
        except:            
            return "Internal server error", 500
            
    except:
        return "Bad request", 400

if __name__ == '__main__':
    app.run(debug=True,host='10.87.60.4',port=5030)