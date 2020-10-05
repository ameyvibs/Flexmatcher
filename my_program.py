import pandas as pd
import flexmatcher

data1 = [['name','height','age'],['Rohit','160','26'],['Amit','150','15'],['Rahul','180','40']]
df1 = pd.DataFrame(data1,columns=data1.pop(0))
data2 = [['a','n','h'],['50','Raj','165'],['30','Ishan','175'],['35','Ayush','175']]
df2 = pd.DataFrame(data2,columns=data2.pop(0))

mapping1 = {'name':'person_name','age':'person_age','height':'person_height'}
mapping2 = {'n':'person_name','a':'person_age','h':'person_height'}

data3 = [['AG','HT','NM'],['70','155','Dilip'],['10','110','Harsh'],['22','175','Kabir']]
df3 = pd.DataFrame(data3,columns=data3.pop(0))

schema = [df1,df2]
mapping = [mapping1,mapping2]

fm = flexmatcher.FlexMatcher(schema,mapping,sample_size=10)
fm.train()
predict_mapping = fm.make_prediction(df3)
print(predict_mapping)