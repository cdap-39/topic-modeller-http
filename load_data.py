import pandas as pd
df = pd.read_json('data.json')


data = df.content.values.tolist()
print(data[:10])