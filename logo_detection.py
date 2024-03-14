from google.cloud import vision
from google.protobuf.json_format import MessageToJson
from tqdm import tqdm
from PIL import Image
import os
import pandas as pd
import base64
import json

'''
The following script requires Google Cloud CLI to be installed on your machine, 
with default Google account set 
and Google Cloud Vision project created 
together with a billing account.

Please inform yourself about the pricing model of Google Cloud Vision. 
First 1000 images scanned per month are free of charge.

'''

# Connect to Google Cloud and get folder to scan
client = vision.ImageAnnotatorClient()
folder = input("Please paste the folder with images to be scanned in the program folder, then enter the folder's name: \n")
dirname = os.path.dirname(__file__)
path = os.path.join(dirname, folder)
filename = ""

# Create df to export
df = pd.DataFrame(
    columns=[
        "brand",
        "logo",
    ]
)

# Check empty folder
num_files = len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))])
assert num_files > 0, "Folder is empty\n"

# Detect logo in images and write logo to csv 
for filename in tqdm(os.listdir(path)):
    try:
        with open(os.path.join(path, filename), 'rb') as image_file:
            # Use Google Cloud Vision to detect logo
            content = image_file.read()
            image = vision.Image(content=content)     
            response = client.logo_detection(image=image)
            json_response = MessageToJson(response._pb) #convert response to json
            j = json.loads(json_response) #convert response to json
            logos = j["logoAnnotations"]
            for logo in logos:
                # Define crop coordinates
                left = logo["boundingPoly"]["vertices"][0]["x"]
                right = logo["boundingPoly"]["vertices"][1]["x"]
                upper = logo["boundingPoly"]["vertices"][0]["y"]
                lower = logo["boundingPoly"]["vertices"][2]["y"]
                # Crop logo
                original_im = Image.open(image_file)
                logo_im = original_im.crop((left, upper, right, lower))
                # Save brand and logo in base64 format to df
                d = {
                    "brand": [logo["description"]],
                    "logo": [base64.b64encode(logo_im.tobytes())],
                }
                df_item = pd.DataFrame(data=d)
                df = df._append(df_item, ignore_index=True)
                # Save csv for every logo written
                filename = f'{folder}_results.csv'
                df.to_csv(filename, index=False) 
    except KeyError:
        pass

print(f"Logo detection completed. Results saved to {filename}\n")