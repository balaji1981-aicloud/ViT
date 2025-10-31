import pandas as pd
import os


path = r"C:\Projects\HR Data Analysis\Survey Summary\Data"


for path,dirr,file in os.walk(path):
    files = file


o_path =r"C:\Projects\HR Data Analysis\Survey Summary\Output"


index_list = ["MWB Index",
"Diversity & Inclusion (D&I) index",
"Management",
"TEC Index",
"Engagement Index",
"Client orientation Index",]


for i,file in enumerate(files):
    data = pd.read_excel(path+"\\"+file)
    nan_indices = data[data["Unnamed: 0"].isnull()].index
    n = nan_indices[-4]
    dt = data.loc[0:n-1]
    ques_comp = list(dt.loc[0][2:])
    dt1 = data.loc[2:n-1].reset_index(drop=True)
    dt1.columns = ["Company","No of respondants"]+ques_comp
    d = {"Company":"","Respondants":"","Questions":"","Response":"","Difference from 2024":""}
    df = pd.DataFrame(d,index=[0])
    df.drop(0,inplace=True)
    for i,j in enumerate(dt1["Company"]):
        # print(j)
        if j not in index_list:
            comp = j
            respon = dt1["No of respondants"][i]
            resplist = list(dt1.loc[i][2:])
            for m in range(0,len(ques_comp),2):
                # print(ques_comp[m])
                d = {"Company":j,"Respondants":respon,"Questions":ques_comp[m],"Response":resplist[m],"Difference from 2024":resplist[m+1]}
                df2 = pd.DataFrame(d,index=[0])
                df = pd.concat([df,df2],ignore_index=True)
    df.to_excel(o_path+"//"+file)
