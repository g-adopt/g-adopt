# Helper python script to generate CITATION.cff file from latest Zenodo release
# see https://citation-file-format.github.io/ for more information

import urllib.request
import json 
import yaml

# Find record id of latest zenodo upload
permanent_link = urllib.request.urlopen("https://zenodo.org/doi/10.5281/zenodo.5644391")  # This is the permanent link to all Zenodo releases 
record_id = permanent_link.geturl().split('/')[-1]  # geturl() returns the redirected link to the latest zenodo release so this returns the record id for the latest upload.

# Get data from zenodo api into json format
link = urllib.request.urlopen(f"https://zenodo.org/api/records/{record_id}")
response = link.read().decode('utf8')
response_json = json.loads(response)

# Create a dictionary with information needed for CITATION.cff format
data = {'cff-version': "1.2.0",
        'message': "If you use this software, please cite it as below.",
        'title': "G-ADOPT",
        'doi': "10.5281/zenodo.5644391",
        'version': response_json['metadata']['version'],
        'type': "software",
        'date-released': response_json['metadata']['publication_date'],
        'repository-code': "https://github.com/g-adopt/g-adopt",
        'authors': []
        }
        
# Loop over the authors listed in the latest Zenodo release and to the dictionary 
# Ideally this would also contain orcid information, but not sure how yet...
creators = response_json['metadata']['creators']
for i in range(len(creators)):
    given, family = creators[i]['name'].split()  # Split full name into given and family name
    data['authors'].append({'family-names': family, 'given-names': given})

# Write CITATION.cff file to disk using yaml formatting as per guidelines
with open('CITATION.cff', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)
