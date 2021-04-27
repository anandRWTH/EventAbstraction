import pandas as pd

import os
import zipfile

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".png"):
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(path, '..')))


zipf = zipfile.ZipFile('output.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir("cluster/og_c2/", zipf)
zipf.close()
'''
df = pd.DataFrame({
    'brand': ['Yum Yum', 'Yum Yum', 'Yum Yum', 'Yum Yum', 'Yum Yum', 'Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
    'style': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'pack', 'pack'],
    'rating': [1, 4, 1, 4, 1, 2, 4, 3.5, 15, 5]
})


df.file_name = "abc"

print(df.file_name)

abc = df.copy()

print(abc.file_name)

#print(df[df['brand']=='Yum Yum']['style'].unique())
	
dff = df.copy()
dff['rating1'] = dff['rating']
dff['style1'] = dff['style']
		
#dff = dff.groupby(['brand'], as_index = False).agg({'rating': 'max', 'rating1': 'min'})
print(type(dff))
temp = dff.groupby(['brand', 'style'], as_index = False)
#print(temp.loc(temp['rating']==1))

idx = temp['rating'].idxmax()
idx1 = dff.groupby(['brand', 'style'], as_index = False)['rating'].idxmin()

print(pd.concat([dff.loc[idx,],dff.loc[idx1,]]))
#dff = df.groupby(['brand'], as_index = False)['rating'].first()
#, as_index = False
#print(dff)

print(df)
keys = list(dff.columns.values)
i1 = df.set_index(keys).index
i2 = dff.set_index(keys).index
print(i1)
print(i2)
print(df[~i1.isin(i2)])


temp_first = df[df['brand']=='Yum Yum'].sort_values('style').drop_duplicates(['rating'], keep='first')
temp_last = df[df['brand']=='Yum Yum'].sort_values('style').drop_duplicates(['rating'], keep='last')

result = pd.concat([temp_first, temp_last])
df[df['brand']=='Yum Yum'] = result.sort_values('style').drop_duplicates()

print(df.dropna())

'''
            #duplicates = [item for item, count in collections.Counter(abstracted_trace).items() if count > 1]
