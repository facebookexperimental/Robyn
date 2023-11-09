import pandas as pd
import json
import requests
import json
import binascii
import io
import pandas as pd
import os
import sys
import subprocess
from urllib.request import urlopen
from bs4 import BeautifulSoup
from PIL import Image
import warnings


hexToPng = lambda fileName, hexData: (
    Image.open(  # Open an image using the PIL library
        io.BytesIO(  # Create an in-memory binary stream
            binascii.unhexlify(hexData)  # Convert hexadecimal data back to binary
        )
    ).save(fileName, "png")  # Save the opened image as a PNG file with the given fileName
)

def asSerialisedFeather(modelData):
    """
    Serializes given model data to a Feather-formatted hex string.

    This function takes model data, converts it to a pandas DataFrame,
    serializes the DataFrame into the Feather binary format for efficient
    storage and retrieval, and then encodes this binary data into a hexadecimal
    string for easy transmission or storage in text-based formats.

    Args:
    modelData: The data to serialize. Can be any structure that pandas can
               convert to a DataFrame (like a dict of lists or list of dicts).

    Returns:
    A hexadecimal string representation of the Feather-formatted data.
    """

    # Create an in-memory bytes buffer
    modelDataFeather = io.BytesIO()
    
    # Convert the input model data into a DataFrame and then to Feather format,
    # writing the binary data into our in-memory buffer
    pd.DataFrame(modelData).to_feather(modelDataFeather)
    
    # Move the buffer position to the start of the stream
    modelDataFeather.seek(0)
    
    # Read the binary Feather data from the buffer
    modelDataBinary = modelDataFeather.read()
    
    # Convert the binary data to a hexadecimal string and return it
    return binascii.hexlify(modelDataBinary).decode()


def pandas_builder(jsondata):
    returndf = pd.DataFrame(jsondata)
    return returndf

def robyn_response(argument,payload=0,api='http://127.0.0.1:9999/{}'):
    #if no api string is provided the function with default to "http://127.0.0.1:9999/{}" i.e. localhost at port 9999
    if(payload==0):
        response = requests.get(api.format(argument))
        respJson = json.loads(response.content.decode('utf-8'))
        return respJson
    else:
        response = requests.post(api.format(argument),data=payload)
        respJson = json.loads(response.content.decode('utf-8'))
        return respJson

def render_spendexposure(InputJson,max_size=(1000, 1500)):
    if len(InputJson['exposure_vars']) > 0:
        for i in InputJson['exposure_vars']:
            image_data = binascii.unhexlify("".join(InputJson['modNLS']['plots'][i]))
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            display(image)

def plot_outputgraphs(OutputJson,argumenttype,graphtype,max_size=(1000, 1500)):
    if(graphtype in ['moo_distrb_plot','moo_cloud_plot']):
        image_data = binascii.unhexlify("".join(OutputJson['convergence']['moo_distrb_plot']))
        image = Image.open(io.BytesIO(image_data))
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        display(image)
    elif(graphtype=='ts_validation_plot'):
        if OutputJson['ts_validation_plot']:
            image_data = binascii.unhexlify("".join(OutputJson['ts_validation_plot']))
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            display(image)
    elif(graphtype in ['allocator']):
        image_data = binascii.unhexlify("".join(argumenttype))
        image = Image.open(io.BytesIO(image_data))
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        display(image)
    else:
        warnings.warn("Graphtype does not exist")

def load_modeldata(sol_id,InputJson,OutputJson):
    select_model = sol_id

    onepagersArgs = {
        "select_model" : select_model,
        "export" : False, # this will create files locally
    }

    # Build the payload for the robyn_onepagers()
    payload = {
        'InputCollect' : json.dumps(InputJson),
        'OutputCollect' : json.dumps(OutputJson),
        "jsonOnepagersArgs": json.dumps(onepagersArgs),
        'dpi' : 100,
        'width' : 15,
        'height' : 20
    }

    # Get response
    onepager = robyn_response('robyn_onepagers',payload=payload)
    return onepager

def create_robyn_directory(path):
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)
        
def writefile(datset,path,sol_id):
    create_robyn_directory(path)
    imagepath = path+sol_id
    out_file = open(imagepath+'.jpg', 'wb')
    out_file.write(datset)
    out_file.close()
    print('Onepager written to path:',imagepath)

def load_onepager(InputJson,OutputJson,path,sol='all',top_pareto=False,write=False,max_size=(1000, 1500)):
    if(top_pareto==True and sol=='all'):
        print('Fetching one pager data for top models')
        for i in range(len(OutputJson['clusters']['models'])):
            sol_id = OutputJson['clusters']['models'][i]['solID']
            onepager = load_modeldata(sol_id,InputJson=InputJson,OutputJson=OutputJson)
            image_data = binascii.unhexlify("".join(onepager))
            if(write==True):
                writefile(datset=image_data,path=path,sol_id=sol_id)
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            display(image)
    elif(top_pareto==False and sol=='all'):
        warnings.warn("Too many one pagers to load, please either select top_pareto=True or just specify a solution id")
    elif(top_pareto==False and sol!='all'):
        if(sol in OutputJson['allSolutions']):
            print('Fetching one pager for specified solution id')
            sol_id = sol
            onepager = load_modeldata(sol_id,InputJson=InputJson,OutputJson=OutputJson)
            image_data = binascii.unhexlify("".join(onepager))
            if(write==True):
                writefile(datset=image_data,path=path,sol_id=sol_id)
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            display(image)
        else:
           warnings.warn("Sepcified solution id does not exist. Please check again") 
        
def write_robynmodel(sol,path,InputJson,OutputJson,OutputModels):
    if(sol in OutputJson['allSolutions']):
        writeArgs = {
        "select_model" : sol,
        "export" : True,
        "dir": path
        }

        # Build the payload for the robyn_write()
        payload = {
            'InputCollect' : json.dumps(InputJson),
            'OutputCollect' : json.dumps(OutputJson),
            'OutputModels' : json.dumps(OutputModels),
            "jsonWriteArgs": json.dumps(writeArgs)
        }

        # Get response
        respJson =  robyn_response('robyn_write',payload=payload)
        print('File written to specified path')


