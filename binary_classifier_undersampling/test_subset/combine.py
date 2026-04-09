import xml.etree.ElementTree as ET
import pandas as pd

files = [
    "test_1/annotations.xml",
    "test_2/annotations.xml",
    "test_3/annotations.xml"
]

data = []

for f in files:
    tree = ET.parse(f)
    root = tree.getroot()

    for image in root.findall('image'):
        name = image.get('name')
        tag = image.find('tag')

        if tag is not None:
            label = tag.get('label')
            data.append([name, label])

df = pd.DataFrame(data, columns=['image', 'label'])

# remove duplicates (important)
df = df.drop_duplicates(subset='image')

df.to_csv('labels.csv', index=False)

print("Combined labels saved as labels.csv")