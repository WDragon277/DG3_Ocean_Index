import pandas as pd
from elasticsearch import Elasticsearch
from model.hrdi_p.model import pred_hrci_model

# Connect to Elasticsearch
es = Elasticsearch('http://121.138.113.16:19202')

# Index name and document type
index_name = 'dgl_idx_expo_pred_lst'
doc_type = '_doc'

# Sample data
data = pred_hrci_model()
df_data = pd.DataFrame(data)
json_data = df_data.to_json(orient = 'records')


if not es.indices.exists(index=index_name):
    # Create the index
    es.indices.create(index=index_name)


# Insert data into Elasticsearch
response = es.index(index=index_name, doc_type=doc_type, body=json_data)

# Check if the insertion was successful
if response['result'] == 'created':
    print('Data inserted successfully.')
else:
    print('Failed to insert data.')