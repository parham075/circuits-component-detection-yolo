from operator import index
from tokenize import String
import pandas as pd 
df = pd.read_csv(r'./confusion_M.csv')
c= 0
def color_rule(val):
    
    
    return ['background-color: red' if isinstance(x,float ) and x > 0 else 'background-color: yellow' for x in val]
    

html_column = df.style.apply(color_rule, axis=1)

html_column.to_excel(r'./new_styled.xlsx')

# df = df.replace(0,"")
# df.to_csv(r'./new_confusion_M.csv', sep=',', mode='a')