import json, re, base64, random, requests, traceback, os, time, shutil, sys
from typing import Any
from flask import Flask, json, request, jsonify
from flask_cors import CORS
from flask_sock import Sock
from PIL import Image, ImageOps
from io import BytesIO

from modules import shared, deepbooru, sd_hijack, postprocessing, script_loading

from modules.assistant import ChatAgent
from copy import copy

# class DockerImage(object):
#     def __init__(self,dictionary):
#         for k,v in dictionary.items():
#             self.k = v
    
#     name: str
#     markdown: str
#     dockerImage: str
#     webapi: str
#     webui: str
#     minGpuRamGb: float
#     minFreeSpaceGb: float
#     minCudaVersion: float

# class DockerImagesResponse(object):
#     default: str
#     images: list[DockerImage]
    
    

import toml
import numpy
import importlib  
# instructPix2pix = importlib.import_module("extensions.instructpix2pix.scripts.instruct-pix2pix", "instruct-pix2pix")
loraModule = importlib.import_module("extensions-builtin.Lora.lora")

backupSysPath = sys.path
sys.path = ["extensions/controlnet"] + sys.path
controlnet = importlib.import_module("extensions.controlnet.scripts.controlnet")
controlnetModules = [
"canny",
"depth",
"depth_leres",
"hed",
"mlsd",
"normal_map",
"openpose",
"openpose_hand",
"clip_vision",
"color",
"pidinet",
"scribble",
"fake_scribble",
"segmentation",
"binary",
]
sys.path = backupSysPath

from langchain.callbacks import get_openai_callback
from pathlib import Path

import modules.txt2img
import modules.img2img
import modules.extras
import modules.scripts
import modules.sd_samplers
import modules.sd_models
from modules.paths import models_path
from datetime import datetime
from pprint import pprint
import sys


PATH_TEMP = "tmp/"
PATH_MODELS = "models/Stable-diffusion/"
PATH_EMBEDDINGS = "embeddings/"
PATH_LORA = "models/Lora/"
PATH_HYPERNETWORKS = "models/hypernetworks/"
PATH_CONTROLNET = "models/ControlNet/"

TOML_MODELS = PATH_MODELS + "_models.toml"
TOML_EMBEDDINGS = PATH_EMBEDDINGS + "_embeddings.toml"
TOML_LORA = PATH_LORA + "_loras.toml"
TOML_HYPERNETWORKS = PATH_HYPERNETWORKS + "_hypernetworks.toml"
TOML_CONTROLNET = PATH_CONTROLNET + "_controlnets.toml"

if not os.path.exists(PATH_TEMP):
    os.makedirs(PATH_TEMP)    
if not os.path.exists(PATH_MODELS):
    os.makedirs(PATH_MODELS)    
if not os.path.exists(PATH_EMBEDDINGS):
    os.makedirs(PATH_EMBEDDINGS)    
if not os.path.exists(PATH_LORA):
    os.makedirs(PATH_LORA)    
if not os.path.exists(PATH_HYPERNETWORKS):
    os.makedirs(PATH_HYPERNETWORKS)    
if not os.path.exists(PATH_CONTROLNET):
    os.makedirs(PATH_CONTROLNET)
if not os.path.exists(TOML_MODELS):
    Path(TOML_MODELS).touch()    
if not os.path.exists(TOML_EMBEDDINGS):
    Path(TOML_EMBEDDINGS).touch()    
if not os.path.exists(TOML_LORA):
    Path(TOML_LORA).touch()    
if not os.path.exists(TOML_HYPERNETWORKS):
    Path(TOML_HYPERNETWORKS).touch()    
if not os.path.exists(TOML_CONTROLNET):
    Path(TOML_CONTROLNET).touch()

# https://flask-sse.readthedocs.io/en/latest/advanced.html#channels

api = Flask(__name__)
api.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}

webapi_secret = os.getenv('WEBAPI_SECRET', None)
if webapi_secret != None:
    print("Starting with Webapi Secret: " + webapi_secret)
else:
    print("Starting without Webapi Secret")

sock = Sock(api)
CORS(api)

is_generating = None
with open(TOML_CONTROLNET, "r+") as f:
    controlnetData = toml.load(f)
file_downloads = {}
next_event_id = 0

def send_update(ws):
    global is_generating
    
    try:
        encoded_image = ""
        if shared.state.current_image:
            buffered = BytesIO()
            shared.state.current_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            img_base64 = bytes("data:image/png;base64,", encoding='utf-8') + img_str
            encoded_image = img_base64.decode("utf-8")

        ws.send(json.dumps({
            "isGenerating": is_generating,
            "image": encoded_image,
            "jobNo": shared.state.job_no,
            "jobCount": shared.state.job_count,
            "samplingStep": shared.state.sampling_step,
            "samplingSteps": shared.state.sampling_steps,
            "fileDownloads": file_downloads,
        }, sort_keys=False))
    except BaseException as err:
        print("error", err)
        print(traceback.format_exc())
        pass
    except: 
        pass

@sock.route('/events')
def echo(ws):
    global next_event_id
    next_event_id += 1
    my_event_id = str(next_event_id)
    try:
        shared.state.register_listener(lambda: send_update(ws), my_event_id)
        while True:
            data = ws.receive().strip()
            if data == 'close':
                break
            elif data == '"status"':
                send_update(ws)
            elif data == '"abort"':
                shared.state.interrupt()
            else:
                print("Unknown WS message: " + data)
    except BaseException as err:
        if '1000' in str(err):
            shared.state.clear_listener(my_event_id)
            return
        print("Exception:")
        pprint(err)
        raise err
    finally:
        shared.state.clear_listener(my_event_id)


def after_run(request_data):
    # shared.state.clear_listeners()
    pass


def before_run(request_data):
    shared.state.sampling_step = 0
    shared.state.job_count = -1
    shared.state.job_no = 0
    shared.state.current_latent = None
    shared.state.current_image = None
    shared.state.current_image_sampling_step = 0
    shared.state.skipped = False
    shared.state.interrupted = False
    shared.state.textinfo = None
    shared.state.job = ""

    args = request.args
    preview = args.get("preview", default="0") == "1"

    if preview:
        shared.opts.show_progress_every_n_steps = 5
    else:
        shared.opts.show_progress_every_n_steps = 0

    dirty = False
    models = modules.sd_models.checkpoint_tiles()
    if 'model' in request_data and request_data['model'] in models and shared.opts.sd_model_checkpoint != request_data[
        'model']:
        print("Loading model: ", request_data['model'])
        shared.opts.sd_model_checkpoint = request_data['model']
        shared.opts.data_labels["sd_model_checkpoint"].onchange()
        dirty = True
    hypernetworks = [x for x in shared.hypernetworks.keys()]

    if 'hypernetwork' in request_data:
        if request_data['hypernetwork'] in hypernetworks and shared.opts.sd_hypernetwork != request_data[
            'hypernetwork']:
            # print("Loading hypernetwork: ", request_data['hypernetwork'])
            shared.opts.sd_hypernetwork = request_data['hypernetwork']
            shared.opts.data_labels["sd_hypernetwork"].onchange()
            dirty = True
        elif not request_data['hypernetwork'] and shared.opts.sd_hypernetwork != "None":
            print("Unloading hypernetwork")
            shared.opts.sd_hypernetwork = "None"
            shared.opts.data_labels["sd_hypernetwork"].onchange()
            dirty = True

    if 'clipIgnoreLastLayers' in request_data and request_data['clipIgnoreLastLayers'] is not None and request_data[
        'clipIgnoreLastLayers'] != shared.opts.CLIP_ignore_last_layers:
        shared.opts.CLIP_ignore_last_layers = request_data['clipIgnoreLastLayers']
        dirty = True

    if dirty:
        shared.opts.save(shared.config_filename)


@api.route('/api/endpoints', methods=['GET'])
def list_endpoints():
    global webapi_secret, controlnetData
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        print("got wrong secret: " + request.headers.get('webapi-secret', "should not happen"))
        return 'wrong secret', 401

    samplers = {}
    for sampler in map(lambda x: x.name, modules.sd_samplers.samplers):
        samplers[sampler] = sampler

    samplers_img2img = {}
    for sampler in map(lambda x: x.name, modules.sd_samplers.samplers_for_img2img):
        samplers_img2img[sampler] = sampler
        
    embeddings = sorted(list(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()), key=str.casefold)
    loras = sorted(list(loraModule.available_loras.keys()), key=str.casefold)
    # print("embeddings: " + str(embeddings))

    upscaleFactor = {
        "min": 1,
        "max": 4,
        "step": 0.5,
        "default": 2
    }

    models = sorted(modules.sd_models.checkpoint_tiles(), key=str.casefold)    
    model = shared.opts.sd_model_checkpoint

    hypernetworks = sorted(list(shared.hypernetworks.keys()), key=str.casefold)
    hypernetworks.sort()
    hypernetwork = shared.opts.sd_hypernetwork
    if hypernetwork == "None":
        hypernetwork = None

    clipIgnoreLastLayers = {
        "min": 0,
        "max": 5,
        "step": 1,
        "default": shared.opts.CLIP_ignore_last_layers
    }

    count = {
        "min": 1,
        "max": 100,
        "step": 1,
        "default": 4
    }
    
    controlnetResolution = {
        "min": 64,
        "max": 2048,
        "step": 1,
        "default": 512
    }
    controlnetThresholdA = {
        "min": 0,
        "max": 255,
        "step": 0.01,
        "default": 100.0
    }
    controlnetThresholdB = {
        "min": 0,
        "max": 255,
        "step": 0.01,
        "default": 200.0
    }
    
    
    with open(TOML_CONTROLNET, "r+") as f:
        controlnetData = toml.load(f)
        
    cn_models = []
    for k, v in controlnet.cn_models.items():
        if v and controlnet.cn_models_dir in v:
            cn_models.append(k)
    controlnetModels = sorted(cn_models)
    
    controlnetWeight = {
        "min": 0.0,
        "max": 2.0,
        "step": 0.05,
        "default": 1.0
    }
    controlnetGuidanceStrength = {
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "default": 1.0
    } 
    controlnetGuidanceStart = {
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "default": 0.0
    } 
    controlnetGuidanceEnd = {
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "default": 1.0
    } 
    latentCoupleEndAtStep = {
        "min": 0.0,
        "max": 150.0,
        "step": 1,
        "default": 20.0        
    }
    
    endpoints = [
        {
            "name": "Text to Image",
            "mode": "txt2img",
            "path": "/api/txt2img",
            "inputs": {
                "count": count,
                "model": model,
                "models": models,
                "hypernetwork": hypernetwork,
                "hypernetworks": hypernetworks,
                "clipIgnoreLastLayers": clipIgnoreLastLayers,
                
                "controlnetModels": controlnetModels,
                "controlnetModules": controlnetModules,
                "controlnetWeight": controlnetWeight,
                "controlnetGuidanceStrength": controlnetGuidanceStrength,
                "controlnetGuidanceStart": controlnetGuidanceStart,
                "controlnetGuidanceEnd": controlnetGuidanceEnd,
                "controlnetResolution": controlnetResolution,
                "controlnetThresholdA": controlnetThresholdA,
                "controlnetThresholdB": controlnetThresholdB,
                
                "latentCouple": True,
                "latentCoupleEndAtStep": latentCoupleEndAtStep,

                "isHighresFix": True,
                "isHighresFixScaleLatent": True,
                "isTiling": True,
                "faceRestoreModels": ['CodeFormer', 'GFPGAN'],
                "faceRestoreStrength": {
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "default": 1
                },

                "prompt": True,
                "negativePrompt": True,
                "sampleSteps": {
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "default": 50
                },
                "guidanceScale": {
                    "min": 1,
                    "max": 50,
                    "step": 0.5,
                    "default": 7.5
                },
                "denoisingStrength": {
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "default": 0.75
                },
                "seed": True,
                "samplers": samplers,
                "sampler": "LMS",
                "imageWidth": {
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "default": 512
                },
                "imageHeight": {
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "default": 512
                },
                # https://formkit.com/advanced/schema
                # "custom": [
                #     # {
                #     #     "$el": 'h1',
                #     #     "children": 'Custom Settings'
                #     # },
                #     # {
                #     #     "$formkit": 'text',
                #     #     "name": 'email',
                #     #     "label": 'Email',
                #     #     "help": 'This will be used for your account.',
                #     #     "validation": 'required|email',
                #     # },
                #     {
                #         "$formkit": 'text',
                #         "name": 'variantSeed',
                #         "label": 'VariantSeed',
                #         # "help": 'This will be used for your account.',
                #         # "validation": 'required'
                #     },
                #     {
                #         "$formkit": 'primeSlider',
                #         "name": 'variantStrength',
                #         "label": 'Variant Strength %',
                #         "min": 0,
                #         "max": 100,
                #         "step": 1,
                #         "value": 80,
                #     },
                # ]
            },
            "outputs": {
                "image": "png",
            }
        },
        {
            "name": "Image to Image",
            "mode": "img2img",
            "path": "/api/img2img",
            "inputs": {
                "count": count,
                "model": model,
                "models": models,
                "hypernetwork": hypernetwork,
                "hypernetworks": hypernetworks,
                "clipIgnoreLastLayers": clipIgnoreLastLayers,
                
                "controlnetModels": controlnetModels,
                "controlnetModules": controlnetModules,
                "controlnetWeight": controlnetWeight,
                "controlnetGuidanceStrength": controlnetGuidanceStrength,
                "controlnetGuidanceStart": controlnetGuidanceStart,
                "controlnetGuidanceEnd": controlnetGuidanceEnd,
                "controlnetResolution": controlnetResolution,
                "controlnetThresholdA": controlnetThresholdA,
                "controlnetThresholdB": controlnetThresholdB,
                
                "latentCouple": True,
                "latentCoupleEndAtStep": latentCoupleEndAtStep,
                
                "faceRestoreModels": ['CodeFormer', 'GFPGAN'],
                "faceRestoreStrength": {
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "default": 1
                },
                "image": True,
                "mask": True,
                "prompt": True,
                "negativePrompt": True,
                "sampleSteps": {
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "default": 50
                },
                "imageResizeMode": "Just resize",
                "imageResizeModes": ["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"],
                "maskContentMode": "fill",
                "maskContentModes": ['fill', 'original', 'latent noise', 'latent nothing'],
                "maskInpaintMode": "Inpaint masked",
                "maskInpaintModes": ['Inpaint masked', 'Inpaint not masked'],
                "maskInpaintFullResolution": True,
                "maskInpaintFullResolutionPadding": {
                    "min": 0,
                    "max": 256,
                    "step": 4,
                    "default": 32   
                },
                "maskBlurStrength": {
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "default": 4
                },
                "guidanceScale": {
                    "min": 1,
                    "max": 50,
                    "step": 0.5,
                    "default": 7.5
                },
                "denoisingStrength": {
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "default": 0.75
                },
                "seed": True,
                "samplers": samplers_img2img,
                "sampler": "LMS",
                "imageWidth": {
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "default": 512
                },
                "imageHeight": {
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "default": 512
                },
            },
            "outputs": {
                "image": "png"
            }
        },
        {
            "name": "Image to Prompt",
            "mode": "img2prompt",
            "path": "/api/img2prompt",
            "inputs": {
                "image": True,
                "interrogator": "CLIP",
                "interrogators": ["CLIP", "DeepBooru"]
            },
            "outputs": {
                "prompt": True
            }
        },
        {
            "name": "Upscale",
            "mode": "upscale",
            "path": "/api/upscale",
            "submodes": [
                {
                    "name": "Lanczos",
                    "submode": "Lanczos",
                    "inputs": {
                        "upscaleFactor": upscaleFactor,
                    }
                },
                {
                    "name": "LDSR",
                    "submode": "LDSR",
                    "inputs": {
                        "upscaleFactor": upscaleFactor,
                    }
                },
                {
                    "name": "ESRGAN 4x",
                    "submode": "ESRGAN 4x",
                    "inputs": {
                        "upscaleFactor": upscaleFactor,
                    }
                },
                {
                    "name": "BSRGAN 4x",
                    "submode": "BSRGAN 4x",
                    "inputs": {
                        "upscaleFactor": upscaleFactor,
                    }
                },
                {
                    "name": "SwinIR 4x",
                    "submode": "SwinIR 4x",
                    "inputs": {
                        "upscaleFactor": upscaleFactor,
                    }
                }
            ],
            "inputs": {
                "image": True,
                'fixFaces': 'codeformer',
                'fixFacesOptions': {
                    'Codeformer': 'codeformer',
                    'GFPGAN': 'gfpgan',
                },
                "fixFacesStrength": {
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "default": 0.9
                }
            },
            "outputs": {
                "image": "png"
            }
        }
    ]    
    
    
    with open(TOML_MODELS, "r+") as f:
        modelData = toml.load(f)
    with open(TOML_EMBEDDINGS, "r+") as f:
        embeddingData = toml.load(f)
    with open(TOML_LORA, "r+") as f:
        loraData = toml.load(f)
    with open(TOML_HYPERNETWORKS, "r+") as f:
        hypernetworkData = toml.load(f)

    
    # for endpoint in endpoints:
    #     if endpoint["mode"] == 'img2img':
    #         depthEndpoint = copy(endpoint)
    #         depthEndpoint["name"] = "Depth to Image"
    #         depthEndpoint["mode"] = "depth2img"
    #         depthEndpoint["inputs"]["model"] = depthModel
    #         depthEndpoint["inputs"]["models"] = [depthModel]
    #         endpoints.insert(2, depthEndpoint)
    
    result=shutil.disk_usage('.')
    return jsonify({
        "endpoints": endpoints, 
        "modelData": modelData, 
        "hypernetworkData": hypernetworkData,
        "embeddings": embeddings,
        "embeddingData": embeddingData, 
        "controlnetData": controlnetData,
        "loras": loras,
        "loraData": loraData,
        "diskSpaceFree": result.free,
        "diskSpaceUsed": result.used,
        "diskSpaceTotal": result.total,
    }), 200

@api.route('/api/reload', methods=['GET'])
def reload():
    global webapi_secret
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401
    try:
        loraModule.list_available_loras()
        shared.reload_hypernetworks()
        shared.refresh_checkpoints()
        if not shared.sd_model:
            modules.sd_models.load_model()
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(True)
        shared.state.interrupt()
        shared.state.need_restart = True
        return {'result': 'OK'}, 200
    except BaseException as err:
        print("error", err)
        print(traceback.format_exc())
        return "Error: {0}".format(err), 500
    
@api.route('/api/entityDelete', methods=['POST'])
def entity_delete():
    global webapi_secret, controlnetData
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401
    if not shared.sd_model:
        return 'still booting up', 500
    
    try:
        args = request.args
        entityType:Any = args.get("entityType")
        entityKey:Any = args.get("entityKey")

        deleteDirectory = None
        deleteFileStart = entityKey
        
        # model, embedding, hypernetwork, lora, controlnet
        if entityType == "model":
            deleteDirectory = PATH_MODELS
            deleteFileStart = Path(entityKey).stem
            targetToml = TOML_MODELS
        elif entityType == "embedding":
            deleteDirectory = PATH_EMBEDDINGS
            targetToml = TOML_EMBEDDINGS
        elif entityType == "hypernetwork":
            deleteDirectory = PATH_HYPERNETWORKS
            targetToml = TOML_HYPERNETWORKS
        elif entityType == "lora":
            deleteDirectory = PATH_LORA
            targetToml = TOML_LORA
        elif entityType == "controlnet":
            deleteDirectory = PATH_CONTROLNET
            deleteFileStart = Path(entityKey).stem
            targetToml = TOML_CONTROLNET
        else:
            return {'result': 'INVALID'}, 500
        
        if deleteDirectory and deleteFileStart:
            for fname in os.listdir(deleteDirectory):
                if fname.startswith(deleteFileStart):
                    fpath = os.path.join(deleteDirectory, fname)
                    print("delete " + str(fpath))
                    os.remove(fpath)
        
        if targetToml:
            with open(targetToml, "r+") as f:
                data = toml.load(f)
                data.pop(entityKey)
                
            with open(targetToml, "w+") as f:
                f.write(toml.dumps(data))
            
            if entityKey == "controlnet":
                controlnetData = data
                
            print("removed " + entityKey + " from " + targetToml)
            
        return {'result': 'OK'}, 200
    except BaseException as err:
        print("error", err)
        print(traceback.format_exc())
        return "Error: {0}".format(err), 500

@api.route('/api/entityUpdate', methods=['POST'])
def entity_update():
    global webapi_secret, controlnetData
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401
    if not shared.sd_model:
        return 'still booting up', 500
    
    try:
        args = request.args
        request_data: Any = request.get_json()
        entityType:Any = args.get("entityType")
        entityKey:Any = args.get("entityKey")
        
        # model, embedding, hypernetwork, lora, controlnet
        if entityType == "model":
            targetToml = TOML_MODELS
        elif entityType == "embedding":
            targetToml = TOML_EMBEDDINGS
        elif entityType == "hypernetwork":
            targetToml = TOML_HYPERNETWORKS
        elif entityType == "lora":
            targetToml = TOML_LORA            
        elif entityType == "controlnet":
            targetToml = TOML_CONTROLNET
        else:
            return {'result': 'INVALID'}, 500
            
        if targetToml:
            with open(targetToml, "r+") as f:
                data = toml.load(f)
                data[entityKey] = request_data
                
            with open(targetToml, "w+") as f:
                f.write(toml.dumps(data))
            
            if entityKey == "controlnet":
                controlnetData = data
                
            print("updated " + entityKey + " in " + targetToml)
        
        return {'result': 'OK'}, 200
    except BaseException as err:
        print("error", err)
        print(traceback.format_exc())
        return "Error: {0}".format(err), 500

    

@api.route('/api/entityDownload', methods=['POST'])
def download():
    global is_generating, webapi_secret, controlnetData
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401
    try:
        request_data: Any = request.get_json()
        queueId = request_data.get("id")
        filename = request_data.get("filename")
        downloadUrl = request_data.get("downloadUrl")
        downloadType = request_data.get("downloadType")
        totalSizeMb = request_data.get("totalSizeMb") 
        name = request_data.get("name") 
        description = request_data.get("description") 
        baseModel = request_data.get("baseModel") 
        sourceUrl = request_data.get("sourceUrl") 
        imageUrl = request_data.get("imageUrl") 
        words = request_data.get("words") 
        category = request_data.get("category")
        
        total_size = int(totalSizeMb * 1024.0 * 1024.0)
        
        start_time = time.time()
        downloaded_size = 0
        speed = 0
        
        tmpPath = PATH_TEMP + filename
        targetPath = None
        targetToml = None
        targetTomlKey = filename
        targetTomlValue = None
                
        nowstr = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z:%M")
        
        # model, embedding, hypernetwork, lora, controlnet
        if downloadType == "model":
            targetPath = PATH_MODELS + filename
            if not filename.endswith('.vae') and not filename.endswith(".yml"):
                targetToml = TOML_MODELS
                targetTomlValue = {
                    "name": name,
                    "description": description,
                    "added": nowstr,
                    "sourceUrl": sourceUrl,
                    "words": words,
                    "category": category,
                    "baseModel": baseModel,
                    "imageUrl": imageUrl,
                }
            
        elif downloadType == "embedding":
            targetPath = PATH_EMBEDDINGS + filename 
            targetToml = TOML_EMBEDDINGS
            targetTomlKey = Path(filename).stem
            targetTomlValue = {
                "name": name,
                "description": description,
                "added": nowstr,
                "sourceUrl": sourceUrl,
                "baseModel": baseModel,
                "category": category,
                "imageUrl": imageUrl,
            }
        elif downloadType == "hypernetwork":
            targetPath = PATH_HYPERNETWORKS + filename 
            targetToml = TOML_HYPERNETWORKS
            targetTomlKey = Path(filename).stem
            targetTomlValue = {
                "name": name,
                "description": description,
                "added": nowstr,
                "sourceUrl": sourceUrl,
                "baseModel": baseModel,
                "words": words,
                "category": category,
                "imageUrl": imageUrl,
            }
            
        elif downloadType == "lora":
            targetPath = PATH_LORA + filename 
            targetToml = TOML_LORA
            targetTomlKey = Path(filename).stem
            targetTomlValue = {
                "name": name,
                "description": description,
                "added": nowstr,
                "sourceUrl": sourceUrl,
                "baseModel": baseModel,
                "words": words,
                "category": category,
                "imageUrl": imageUrl,
            }
            
        elif downloadType == "controlnet":
            targetPath = PATH_CONTROLNET + filename 
            targetToml = TOML_CONTROLNET
            targetTomlValue = {
                "name": name,
                "description": description,
                "added": nowstr,
                "sourceUrl": sourceUrl,
                "category": category,
                "modules": words,
                "baseModel": baseModel,
                "imageUrl": imageUrl,
            }
        else:
            return {'result': 'INVALID'}, 500
        
        def send_download_update(status, errorMessage = None):
            file_downloads[queueId] = {
                'totalSizeMb': total_size / 1024.0 / 1024.0,
                'downloadedSizeMb': downloaded_size / 1024.0 / 1024.0,
                'downloadSpeedMbits': speed / (1024 * 1024) * 8,
                'status': status,
                'errorMessage': errorMessage
            }
            shared.state.call_listeners()
        send_download_update('downloading')
        last_download_update = time.time()
        
        try:
            with open(tmpPath, 'wb') as f:
                for chunk in requests.get(downloadUrl, stream=True).iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        f.flush()                  
                        downloaded_size += len(chunk)      
                        now = time.time()
                        if now - last_download_update > 1.0:
                            last_download_update = now
                            speed = calculate_download_speed(downloaded_size, start_time)
                            progress = downloaded_size / total_size * 100
                            print("downloading progress " + "{:.2f}".format(progress) + "%")
                            send_download_update('downloading')
            print("download done at " + tmpPath)
            shutil.move(tmpPath, targetPath)
            print("download moved to " + targetPath)
            
            if targetToml and targetTomlValue:                 
                with open(targetToml) as f:
                    data = toml.load(f)
                    data[targetTomlKey] = targetTomlValue
                    
                with open(targetToml, "w") as f:
                    f.write(toml.dumps(data))
                    print("saved " + targetToml)
                
                if downloadType == "controlnet":
                    controlnetData = data
                        
            send_download_update('done')
            file_downloads.pop(queueId)
            return {'result': 'OK'}, 200
        except BaseException as err:
            print("error", err)
            print(traceback.format_exc())
            send_download_update('error', "Error: {0}".format(err))
            file_downloads.pop(queueId)
            return {'result': 'ERROR'}, 500
        except:
            send_download_update('error')
            file_downloads.pop(queueId)
            return {'result': 'ERROR'}, 500
            
    except BaseException as err:
        print("error", err)
        print(traceback.format_exc())
        return "Error: {0}".format(err), 500
        


def calculate_download_speed(downloaded_size, start_time):
    """
    A function to calculate the download speed in bytes per second.
    """
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        return downloaded_size / elapsed_time
    else:
        return 0

def get_controlnet_model(name):
    global controlnetData
    for k,v in controlnetData.items():
        if v["module"] == name:
            return k
    return 'None'

    
@api.route('/api/txt2img', methods=['POST'])
def txt2img():
    global is_generating, webapi_secret, controlnetData
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401
    if not shared.sd_model:
        return 'still booting up', 500
    if is_generating:
        print("already generating " + is_generating)
        return 'already generating', 500
    try:
        args = request.args
        dreamId = args.get("dreamId", default="none")
        is_generating = dreamId
        request_data: Any = request.get_json()
        # print(request_data)

        samplers = []
        for sampler in map(lambda x: x.name, modules.sd_samplers.samplers):
            samplers.append(sampler)

        prompt = request_data["prompt"] if "prompt" in request_data else ""
        negative_prompt = request_data["negativePrompt"] if "negativePrompt" in request_data else ""
        prompt_styles = []
        steps = request_data["sampleSteps"] if "sampleSteps" in request_data else 10
        sampler = request_data["sampler"] if "sampler" in request_data else "LMS"
        sampler_index = samplers.index(sampler) if sampler in samplers else 0
        
        faceRestoreModel = request_data["faceRestoreModel"] if "faceRestoreModel" in request_data else ""
        restore_faces = True if faceRestoreModel else False
        if restore_faces:
            shared.opts.face_restoration_model = faceRestoreModel
            shared.opts.code_former_weight = request_data["faceRestoreStrength"] if "faceRestoreStrength" in request_data else 1
        
        tiling = request_data["isTiling"] if "isTiling" in request_data else False
        n_iter = 1
        batch_size = 1
        cfg_scale = request_data["guidanceScale"] if "guidanceScale" in request_data else 7.5
        seed = seed_to_int(request_data["seed"] if "seed" in request_data else "")
        subseed = 0
        subseed_strength = 0
        seed_resize_from_h = 0
        seed_resize_from_w = 0
        seed_enable_extras = False
        height = request_data["height"] if "height" in request_data else 512
        width = request_data["width"] if "width" in request_data else 512
        enable_hr = request_data["isHighresFix"] if "isHighresFix" in request_data else False
        # scale_latent = request_data["isHighresFixScaleLatent"] if "isHighresFixScaleLatent" in request_data else False
        denoising_strength = request_data["denoisingStrength"] if "denoisingStrength" in request_data else 0.75
        script_args = 0
        hr_scale = 0
        hr_upscaler = None
        hr_second_pass_steps = 0
        hr_resize_x = 0
        hr_resize_y = 0
        id_task = dreamId 
        override_settings_texts = []
        controlnet1_module = request_data["controlnet1Module"] if "controlnet1Module" in request_data else "none"
        
        
        controlnet1_model = request_data["controlnet1Model"] if "controlnet1Model" in request_data else get_controlnet_model(controlnet1_module)
        if "controlnet1Preprocess" in request_data and not request_data['controlnet1Preprocess']:
            controlnet1_module = "none"
        controlnet1_weight = request_data["controlnet1Weight"] if "controlnet1Weight" in request_data else 1
        controlnet1_guidance_start = request_data["controlnet1GuidanceStart"] if "controlnet1GuidanceStart" in request_data else 0.0
        controlnet1_guidance_end = request_data["controlnet1GuidanceEnd"] if "controlnet1GuidanceEnd" in request_data else 1.0
        controlnet1_scribble_mode = request_data["controlnet1Scribble"] if "controlnet1Scribble" in request_data else False
        controlnet1_rgb_to_bgr_mode = request_data["controlnet1RgbToBgr"] if "controlnet1RgbToBgr" in request_data else False
        controlnet1_low_vram_mode = False
        controlnet1_guess_mode = request_data["controlnet1GuessMode"] if "controlnet1GuessMode" in request_data else False
        controlnet1_resize_mode = request_data["controlnet1ResizeMode"] if "controlnet1ResizeMode" in request_data else "Envelope (Outer Fit)"
        controlnet1_resolution = request_data["controlnet1Resolution"] if "controlnet1Resolution" in request_data else 512
        controlnet1_threshold_a = request_data["controlnet1ThresholdA"] if "controlnet1ThresholdA" in request_data else 0.1
        controlnet1_threshold_b = request_data["controlnet1ThresholdB"] if "controlnet1ThresholdB" in request_data else 0.1
        
        controlnet2_module = request_data["controlnet2Module"] if "controlnet2Module" in request_data else "none"
        controlnet2_model = request_data["controlnet2Model"] if "controlnet2Model" in request_data else get_controlnet_model(controlnet2_module)
        if "controlnet2Preprocess" in request_data and not request_data['controlnet2Preprocess']:
            controlnet2_module = "none"
        controlnet2_weight = request_data["controlnet2Weight"] if "controlnet2Weight" in request_data else 1
        controlnet2_guidance_start = request_data["controlnet2GuidanceStart"] if "controlnet2GuidanceStart" in request_data else 0.0
        controlnet2_guidance_end = request_data["controlnet2GuidanceEnd"] if "controlnet2GuidanceEnd" in request_data else 1.0
        controlnet2_scribble_mode = request_data["controlnet2Scribble"] if "controlnet2Scribble" in request_data else False
        controlnet2_rgb_to_bgr_mode = request_data["controlnet2RgbToBgr"] if "controlnet2RgbToBgr" in request_data else False
        controlnet2_low_vram_mode = False
        controlnet2_guess_mode = request_data["controlnet2GuessMode"] if "controlnet2GuessMode" in request_data else False
        controlnet2_resize_mode = request_data["controlnet2ResizeMode"] if "controlnet2ResizeMode" in request_data else "Envelope (Outer Fit)"
        controlnet2_resolution = request_data["controlnet2Resolution"] if "controlnet2Resolution" in request_data else 512
        controlnet2_threshold_a = request_data["controlnet2ThresholdA"] if "controlnet2ThresholdA" in request_data else 0.1
        controlnet2_threshold_b = request_data["controlnet2ThresholdB"] if "controlnet2ThresholdB" in request_data else 0.1
        
        controlnet3_module = request_data["controlnet3Module"] if "controlnet3Module" in request_data else "none"
        controlnet3_model = request_data["controlnet3Model"] if "controlnet3Model" in request_data else get_controlnet_model(controlnet3_module)
        if "controlnet3Preprocess" in request_data and not request_data['controlnet3Preprocess']:
            controlnet3_module = "none"
        controlnet3_weight = request_data["controlnet3Weight"] if "controlnet3Weight" in request_data else 1
        controlnet3_guidance_start = request_data["controlnet3GuidanceStart"] if "controlnet3GuidanceStart" in request_data else 0.0
        controlnet3_guidance_end = request_data["controlnet3GuidanceEnd"] if "controlnet3GuidanceEnd" in request_data else 1.0
        controlnet3_scribble_mode = request_data["controlnet3Scribble"] if "controlnet3Scribble" in request_data else False
        controlnet3_rgb_to_bgr_mode = request_data["controlnet3RgbToBgr"] if "controlnet3RgbToBgr" in request_data else False
        controlnet3_low_vram_mode = False
        controlnet3_guess_mode = request_data["controlnet3GuessMode"] if "controlnet3GuessMode" in request_data else False
        controlnet3_resize_mode = request_data["controlnet3ResizeMode"] if "controlnet3ResizeMode" in request_data else "Envelope (Outer Fit)"
        controlnet3_resolution = request_data["controlnet3Resolution"] if "controlnet3Resolution" in request_data else 512
        controlnet3_threshold_a = request_data["controlnet3ThresholdA"] if "controlnet3ThresholdA" in request_data else 0.1
        controlnet3_threshold_b = request_data["controlnet3ThresholdB"] if "controlnet3ThresholdB" in request_data else 0.1
        
        
        controlnet1_image = None
        controlnet1_inputImage = request_data["controlnet1Image"] if 'controlnet1Image' in request_data else None
        if controlnet1_inputImage:
            if controlnet1_inputImage.startswith('data:'):
                base64_data = re.sub('^data:image/.+;base64,', '', controlnet1_inputImage)
                im_bytes = base64.b64decode(base64_data)
                image_data = BytesIO(im_bytes)
                controlnet_img = Image.open(image_data).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet1_image = {'image': controlnet_img2, 'mask': numpy.zeros(controlnet_img2.shape, dtype=numpy.uint8)}
            else:
                print("downloading controlnet1Image from " + controlnet1_inputImage)
                response = requests.get(controlnet1_inputImage)
                controlnet_img =  Image.open(BytesIO(response.content)).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet1_image = {'image': controlnet_img2, 'mask': numpy.zeros(controlnet_img2.shape, dtype=numpy.uint8)}
            
        controlnet1_maskImage = request_data["controlnet1MaskImage"] if 'controlnet1MaskImage' in request_data else None
        if controlnet1_maskImage and controlnet1_image:
            if controlnet1_maskImage.startswith('data:'):
                base64_data = re.sub('^data:image/.+;base64,', '', controlnet1_maskImage)
                im_bytes = base64.b64decode(base64_data)
                image_data = BytesIO(im_bytes)
                controlnet_img = Image.open(image_data).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet1_image['mask'] = controlnet_img2
            else:
                print("downloading controlnet1_maskImage from " + controlnet1_maskImage)
                response = requests.get(controlnet1_maskImage)
                controlnet_img =  Image.open(BytesIO(response.content)).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet1_image['mask'] = controlnet_img2
        
        controlnet1_is_enabled = True if controlnet1_image else False
        
        controlnet2_image = None
        controlnet2_inputImage = request_data["controlnet2Image"] if 'controlnet2Image' in request_data else None
        if controlnet2_inputImage:
            if controlnet2_inputImage.startswith('data:'):
                base64_data = re.sub('^data:image/.+;base64,', '', controlnet2_inputImage)
                im_bytes = base64.b64decode(base64_data)
                image_data = BytesIO(im_bytes)
                controlnet_img = Image.open(image_data).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet2_image = {'image': controlnet_img2, 'mask': numpy.zeros(controlnet_img2.shape, dtype=numpy.uint8)}
            else:
                print("downloading controlnet2Image from " + controlnet2_inputImage)
                response = requests.get(controlnet2_inputImage)
                controlnet_img =  Image.open(BytesIO(response.content)).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet2_image = {'image': controlnet_img2, 'mask': numpy.zeros(controlnet_img2.shape, dtype=numpy.uint8)}
            
        controlnet2_maskImage = request_data["controlnet2MaskImage"] if 'controlnet2MaskImage' in request_data else None
        if controlnet2_maskImage and controlnet2_image:
            if controlnet2_maskImage.startswith('data:'):
                base64_data = re.sub('^data:image/.+;base64,', '', controlnet2_maskImage)
                im_bytes = base64.b64decode(base64_data)
                image_data = BytesIO(im_bytes)
                controlnet_img = Image.open(image_data).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet2_image['mask'] = controlnet_img2
            else:
                print("downloading controlnet2_maskImage from " + controlnet2_maskImage)
                response = requests.get(controlnet2_maskImage)
                controlnet_img =  Image.open(BytesIO(response.content)).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet2_image['mask'] = controlnet_img2
        
        controlnet2_is_enabled = True if controlnet2_image else False
        
        controlnet3_image = None
        controlnet3_inputImage = request_data["controlnet3Image"] if 'controlnet3Image' in request_data else None
        if controlnet3_inputImage:
            if controlnet3_inputImage.startswith('data:'):
                base64_data = re.sub('^data:image/.+;base64,', '', controlnet3_inputImage)
                im_bytes = base64.b64decode(base64_data)
                image_data = BytesIO(im_bytes)
                controlnet_img = Image.open(image_data).convert('RGB')
                controlnet_img3 = numpy.array(controlnet_img)
                controlnet3_image = {'image': controlnet_img3, 'mask': numpy.zeros(controlnet_img3.shape, dtype=numpy.uint8)}
            else:
                print("downloading controlnet3Image from " + controlnet3_inputImage)
                response = requests.get(controlnet3_inputImage)
                controlnet_img =  Image.open(BytesIO(response.content)).convert('RGB')
                controlnet_img3 = numpy.array(controlnet_img)
                controlnet3_image = {'image': controlnet_img3, 'mask': numpy.zeros(controlnet_img3.shape, dtype=numpy.uint8)}
            
        controlnet3_maskImage = request_data["controlnet3MaskImage"] if 'controlnet3MaskImage' in request_data else None
        if controlnet3_maskImage and controlnet3_image:
            if controlnet3_maskImage.startswith('data:'):
                base64_data = re.sub('^data:image/.+;base64,', '', controlnet3_maskImage)
                im_bytes = base64.b64decode(base64_data)
                image_data = BytesIO(im_bytes)
                controlnet_img = Image.open(image_data).convert('RGB')
                controlnet_img3 = numpy.array(controlnet_img)
                controlnet3_image['mask'] = controlnet_img3
            else:
                print("downloading controlnet3_maskImage from " + controlnet3_maskImage)
                response = requests.get(controlnet3_maskImage)
                controlnet_img =  Image.open(BytesIO(response.content)).convert('RGB')
                controlnet_img3 = numpy.array(controlnet_img)
                controlnet3_image['mask'] = controlnet_img3
        
        controlnet3_is_enabled = True if controlnet3_image else False
        
        latent_couple_enabled = request_data["latentCoupleEnabled"] if 'latentCoupleEnabled' in request_data else False
        latent_couple_divisions = request_data["latentCoupleDivisions"] if 'latentCoupleDivisions' in request_data else "1:1,1:2,1:2"
        latent_couple_positions = request_data["latentCouplePositions"] if 'latentCouplePositions' in request_data else "0:0,0:0,0:1"
        latent_couple_weights = request_data["latentCoupleWeights"] if 'latentCoupleWeights' in request_data else "0.2,0.8,0.8"
        latent_couple_end_at_step = request_data["latentCoupleEndAtStep"] if 'latentCoupleEndAtStep' in request_data else 20
        composable_lora_enabled = latent_couple_enabled
        composable_lora_text_model_encoder = False
        composable_lora_diffusion_model = False

        # txt2img(id_task: str, prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, steps: int,
        # sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float,
        # seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int,
        # seed_enable_extras: bool, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float,
        # hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, *args

        before_run(request_data)
        images, generation_info_js, stats, comments = modules.txt2img.txt2img(
            id_task, prompt, negative_prompt, prompt_styles, steps, sampler_index, restore_faces, 
            tiling, n_iter, batch_size,  cfg_scale, seed, subseed, subseed_strength, 
            seed_resize_from_h, seed_resize_from_w, seed_enable_extras, height, width, enable_hr,
            denoising_strength, hr_scale, hr_upscaler, hr_second_pass_steps, hr_resize_x, 
            hr_resize_y, override_settings_texts, 
            script_args, 
            False,
            False,
            controlnet1_is_enabled, 
            controlnet1_module, 
            controlnet1_model, 
            controlnet1_weight, 
            controlnet1_image,
            controlnet1_scribble_mode, 
            controlnet1_resize_mode, 
            controlnet1_rgb_to_bgr_mode, 
            controlnet1_low_vram_mode, 
            controlnet1_resolution,
            controlnet1_threshold_a, 
            controlnet1_threshold_b,
            controlnet1_guidance_start,
            controlnet1_guidance_end,
            controlnet1_guess_mode,
                       
            
            controlnet2_is_enabled, 
            controlnet2_module, 
            controlnet2_model, 
            controlnet2_weight, 
            controlnet2_image,
            controlnet2_scribble_mode, 
            controlnet2_resize_mode, 
            controlnet2_rgb_to_bgr_mode, 
            controlnet2_low_vram_mode, 
            controlnet2_resolution,
            controlnet2_threshold_a, 
            controlnet2_threshold_b,
            controlnet2_guidance_start,
            controlnet2_guidance_end,
            controlnet2_guess_mode,
            
            controlnet3_is_enabled, 
            controlnet3_module, 
            controlnet3_model, 
            controlnet3_weight, 
            controlnet3_image,
            controlnet3_scribble_mode, 
            controlnet3_resize_mode, 
            controlnet3_rgb_to_bgr_mode, 
            controlnet3_low_vram_mode, 
            controlnet3_resolution,
            controlnet3_threshold_a, 
            controlnet3_threshold_b,
            controlnet3_guidance_start,
            controlnet3_guidance_end,
            controlnet3_guess_mode,
            
            composable_lora_enabled,
            composable_lora_text_model_encoder,
            composable_lora_diffusion_model,
            
            latent_couple_enabled,
            latent_couple_divisions,
            latent_couple_positions,
            latent_couple_weights,
            latent_couple_end_at_step,
)
        
        after_run(request_data)
        encoded_image = None
        second_encoded_image = None
        third_encoded_image = None
        fourth_encoded_image = None
        for image in images:
            buffered = BytesIO()
            if hasattr(image, 'save'):
                image.save(buffered, format="PNG")
            else:
                pil_image = Image.fromarray(image)
                pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            img_base64 = bytes("data:image/png;base64,", encoding='utf-8') + img_str
            
            if not encoded_image:
                encoded_image = img_base64.decode("utf-8")
            elif not second_encoded_image:
                second_encoded_image = img_base64.decode("utf-8")
            elif not third_encoded_image:
                third_encoded_image = img_base64.decode("utf-8")
            elif not fourth_encoded_image:
                fourth_encoded_image = img_base64.decode("utf-8")
            else:
                break
        if shared.state.interrupted:
            return "aborted", 500
        return {
            "generatedImage": encoded_image,
            "generatedSecondImage": second_encoded_image,
            "generatedThirdImage": third_encoded_image,
            "generatedFourthImage": fourth_encoded_image,
            "seed": str(seed),
            "stats": stats,
            "comments": comments,
        }
    except BaseException as err:
        print("error", err)
        print(traceback.format_exc())
        return "Error: {0}".format(err), 500
    finally:
        is_generating = None

@api.route('/api/img2img', methods=['POST'])
def img2img():
    global is_generating, webapi_secret, controlnetData
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401
    if not shared.sd_model:
        return 'still booting up', 500
    if is_generating:
        print("already generating " + is_generating)
        return 'already generating', 500
    try:
        args = request.args
        dreamId = args.get("dreamId", default="none")
        is_generating = dreamId
        request_data: Any = request.get_json()

        samplers = []
        for sampler in map(lambda x: x.name, modules.sd_samplers.samplers_for_img2img):
            samplers.append(sampler)

        init_img = None
        inputImage = request_data["inputImage"]
        if inputImage.startswith('data:'):
            base64_data = re.sub('^data:image/.+;base64,', '', inputImage)
            im_bytes = base64.b64decode(base64_data)
            image_data = BytesIO(im_bytes)
            init_img = Image.open(image_data).convert('RGBA')
        else:
            print("downloading inputImage from " + inputImage)
            response = requests.get(inputImage)
            init_img = Image.open(BytesIO(response.content)).convert('RGBA')
            
        # init_img.save("debug_input.png", format="PNG")

        mode = 0
        init_img_with_mask = None

        if "maskImage" in request_data:
            maskImage = request_data["maskImage"]
            if maskImage != "":
                if maskImage.startswith('data:'):
                    base64_data = re.sub('^data:image/.+;base64,', '', maskImage)
                    im_bytes = base64.b64decode(base64_data)
                    image_data = BytesIO(im_bytes)
                    mask_info = Image.open(image_data)
                else:
                    print("downloading " + maskImage)
                    response = requests.get(maskImage)
                    mask_info = Image.open(BytesIO(response.content))
                mask_info = mask_info.resize(init_img.size).convert('RGBA')
                
                # black_alpha = Image.new("RGBA", mask_info.size, (0, 0, 0, 0))

                # Replace the alpha channel of the original image with the white alpha channel
                # init_img.putalpha(black_alpha.split()[-1])
                mask_info.save("debug_mask.png", format="PNG")
                init_img_with_mask = {"image": init_img, "mask": mask_info}
                init_img = None
                mode = 2
                 
        prompt = request_data["prompt"] if "prompt" in request_data else ""
        negative_prompt = request_data["negativePrompt"] if "negativePrompt" in request_data else ""
        prompt_styles = []
        # init_img = None
        # init_img_with_mask = None
        init_img_inpaint = None
        init_mask_inpaint = None
        inpaint_color_sketch = None
        inpaint_color_sketch_orig = None
        resize_mode = 0
        steps = request_data["sampleSteps"] if "sampleSteps" in request_data else 10
        sampler = request_data["sampler"] if "sampler" in request_data else "LMS"
        sampler_index = samplers.index(sampler) if sampler in samplers else 0
        mask_blur = request_data["maskBlurStrength"] if "maskBlurStrength" in request_data else 4
        mask_alpha = 0
        inpainting_fill = 0
        
        faceRestoreModel = request_data["faceRestoreModel"] if "faceRestoreModel" in request_data else ""
        restore_faces = True if faceRestoreModel else False
        if restore_faces:
            shared.opts.face_restoration_model = faceRestoreModel
            shared.opts.code_former_weight = request_data["faceRestoreStrength"] if "faceRestoreStrength" in request_data else 1
        
        tiling = request_data["isTiling"] if "isTiling" in request_data else False
        n_iter = 1
        inpaint_full_res = request_data["maskInpaintFullResolution"] if "maskInpaintFullResolution" in request_data else False
        inpaint_full_res_padding = request_data["maskInpaintFullResolutionPadding"] if "maskInpaintFullResolutionPadding" in request_data else 0
        inpainting_mask_invert = request_data["maskInpaintMode"] == "Inpaint masked" if "maskInpaintMode" in request_data else False
        batch_size = 1
        cfg_scale = request_data["guidanceScale"] if "guidanceScale" in request_data else 7.5
        seed = seed_to_int(request_data["seed"] if "seed" in request_data else "")
        subseed = 0
        subseed_strength = 0
        seed_resize_from_h = 0
        seed_resize_from_w = 0
        seed_enable_extras = False
        sketch = None
        img2img_batch_input_dir = ""
        img2img_batch_output_dir = ""
        img2img_batch_inpaint_mask_dir = ""
        height = request_data["height"] if "height" in request_data else 512
        width = request_data["width"] if "width" in request_data else 512
        denoising_strength = request_data["denoisingStrength"] if "denoisingStrength" in request_data else 0.75
        script_args = 0
        id_task = dreamId
        image_cfg_scale = 1.5
        
        override_settings_texts = []
        
        
        controlnet1_module = request_data["controlnet1Module"] if "controlnet1Module" in request_data else "none"
        controlnet1_model = get_controlnet_model(controlnet1_module)
        if "controlnet1Preprocess" in request_data and not request_data['controlnet1Preprocess']:
            controlnet1_module = "none"
        controlnet1_weight = request_data["controlnet1Weight"] if "controlnet1Weight" in request_data else 1
        controlnet1_guidance_start = request_data["controlnet1GuidanceStart"] if "controlnet1GuidanceStart" in request_data else 1.0
        controlnet1_guidance_end = request_data["controlnet1GuidanceEnd"] if "controlnet1GuidanceEnd" in request_data else 1.0
        controlnet1_scribble_mode = request_data["controlnet1Scribble"] if "controlnet1Scribble" in request_data else False
        controlnet1_rgb_to_bgr_mode = request_data["controlnet1RgbToBgr"] if "controlnet1RgbToBgr" in request_data else False
        controlnet1_low_vram_mode = False
        controlnet1_guess_mode = request_data["controlnet1GuessMode"] if "controlnet1GuessMode" in request_data else False
        controlnet1_resize_mode = request_data["controlnet1ResizeMode"] if "controlnet1ResizeMode" in request_data else "Envelope (Outer Fit)"
        controlnet1_resolution = request_data["controlnet1Resolution"] if "controlnet1Resolution" in request_data else 512
        controlnet1_threshold_a = request_data["controlnet1ThresholdA"] if "controlnet1ThresholdA" in request_data else 0.1
        controlnet1_threshold_b = request_data["controlnet1ThresholdB"] if "controlnet1ThresholdB" in request_data else 0.1
        
        controlnet2_module = request_data["controlnet2Module"] if "controlnet2Module" in request_data else "none"
        controlnet2_model = get_controlnet_model(controlnet2_module)
        if "controlnet2Preprocess" in request_data and not request_data['controlnet2Preprocess']:
            controlnet2_module = "none"
        controlnet2_weight = request_data["controlnet2Weight"] if "controlnet2Weight" in request_data else 1
        controlnet2_guidance_start = request_data["controlnet2GuidanceStart"] if "controlnet2GuidanceStart" in request_data else 0.0
        controlnet2_guidance_end = request_data["controlnet2GuidanceEnd"] if "controlnet2GuidanceEnd" in request_data else 1.0
        controlnet2_scribble_mode = request_data["controlnet2Scribble"] if "controlnet2Scribble" in request_data else False
        controlnet2_rgb_to_bgr_mode = request_data["controlnet2RgbToBgr"] if "controlnet2RgbToBgr" in request_data else False
        controlnet2_low_vram_mode = False
        controlnet2_guess_mode = request_data["controlnet2GuessMode"] if "controlnet2GuessMode" in request_data else False
        controlnet2_resize_mode = request_data["controlnet2ResizeMode"] if "controlnet2ResizeMode" in request_data else "Envelope (Outer Fit)"
        controlnet2_resolution = request_data["controlnet2Resolution"] if "controlnet2Resolution" in request_data else 512
        controlnet2_threshold_a = request_data["controlnet2ThresholdA"] if "controlnet2ThresholdA" in request_data else 0.1
        controlnet2_threshold_b = request_data["controlnet2ThresholdB"] if "controlnet2ThresholdB" in request_data else 0.1
        
        controlnet3_module = request_data["controlnet3Module"] if "controlnet3Module" in request_data else "none"
        controlnet3_model = get_controlnet_model(controlnet3_module)
        if "controlnet3Preprocess" in request_data and not request_data['controlnet3Preprocess']:
            controlnet3_module = "none"
        controlnet3_weight = request_data["controlnet3Weight"] if "controlnet3Weight" in request_data else 1
        controlnet3_guidance_start = request_data["controlnet3GuidanceStart"] if "controlnet3GuidanceStart" in request_data else 0.0
        controlnet3_guidance_end = request_data["controlnet3GuidanceEnd"] if "controlnet3GuidanceEnd" in request_data else 1.0
        controlnet3_scribble_mode = request_data["controlnet3Scribble"] if "controlnet3Scribble" in request_data else False
        controlnet3_rgb_to_bgr_mode = request_data["controlnet3RgbToBgr"] if "controlnet3RgbToBgr" in request_data else False
        controlnet3_low_vram_mode = False
        controlnet3_guess_mode = request_data["controlnet3GuessMode"] if "controlnet3GuessMode" in request_data else False
        controlnet3_resize_mode = request_data["controlnet3ResizeMode"] if "controlnet3ResizeMode" in request_data else "Envelope (Outer Fit)"
        controlnet3_resolution = request_data["controlnet3Resolution"] if "controlnet3Resolution" in request_data else 512
        controlnet3_threshold_a = request_data["controlnet3ThresholdA"] if "controlnet3ThresholdA" in request_data else 0.1
        controlnet3_threshold_b = request_data["controlnet3ThresholdB"] if "controlnet3ThresholdB" in request_data else 0.1
        
        
        controlnet1_image = None
        controlnet1_inputImage = request_data["controlnet1Image"] if 'controlnet1Image' in request_data else None
        if controlnet1_inputImage:
            if controlnet1_inputImage.startswith('data:'):
                base64_data = re.sub('^data:image/.+;base64,', '', controlnet1_inputImage)
                im_bytes = base64.b64decode(base64_data)
                image_data = BytesIO(im_bytes)
                controlnet_img = Image.open(image_data).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet1_image = {'image': controlnet_img2, 'mask': numpy.zeros(controlnet_img2.shape, dtype=numpy.uint8)}
            else:
                print("downloading controlnet1Image from " + controlnet1_inputImage)
                response = requests.get(controlnet1_inputImage)
                controlnet_img =  Image.open(BytesIO(response.content)).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet1_image = {'image': controlnet_img2, 'mask': numpy.zeros(controlnet_img2.shape, dtype=numpy.uint8)}
            
        controlnet1_maskImage = request_data["controlnet1MaskImage"] if 'controlnet1MaskImage' in request_data else None
        if controlnet1_maskImage and controlnet1_image:
            if controlnet1_maskImage.startswith('data:'):
                base64_data = re.sub('^data:image/.+;base64,', '', controlnet1_maskImage)
                im_bytes = base64.b64decode(base64_data)
                image_data = BytesIO(im_bytes)
                controlnet_img = Image.open(image_data).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet1_image['mask'] = controlnet_img2
            else:
                print("downloading controlnet1_maskImage from " + controlnet1_maskImage)
                response = requests.get(controlnet1_maskImage)
                controlnet_img =  Image.open(BytesIO(response.content)).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet1_image['mask'] = controlnet_img2
        
        controlnet1_is_enabled = True if controlnet1_image else False
        
        controlnet2_image = None
        controlnet2_inputImage = request_data["controlnet2Image"] if 'controlnet2Image' in request_data else None
        if controlnet2_inputImage:
            if controlnet2_inputImage.startswith('data:'):
                base64_data = re.sub('^data:image/.+;base64,', '', controlnet2_inputImage)
                im_bytes = base64.b64decode(base64_data)
                image_data = BytesIO(im_bytes)
                controlnet_img = Image.open(image_data).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet2_image = {'image': controlnet_img2, 'mask': numpy.zeros(controlnet_img2.shape, dtype=numpy.uint8)}
            else:
                print("downloading controlnet2Image from " + controlnet2_inputImage)
                response = requests.get(controlnet2_inputImage)
                controlnet_img =  Image.open(BytesIO(response.content)).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet2_image = {'image': controlnet_img2, 'mask': numpy.zeros(controlnet_img2.shape, dtype=numpy.uint8)}
            
        controlnet2_maskImage = request_data["controlnet2MaskImage"] if 'controlnet2MaskImage' in request_data else None
        if controlnet2_maskImage and controlnet2_image:
            if controlnet2_maskImage.startswith('data:'):
                base64_data = re.sub('^data:image/.+;base64,', '', controlnet2_maskImage)
                im_bytes = base64.b64decode(base64_data)
                image_data = BytesIO(im_bytes)
                controlnet_img = Image.open(image_data).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet2_image['mask'] = controlnet_img2
            else:
                print("downloading controlnet2_maskImage from " + controlnet2_maskImage)
                response = requests.get(controlnet2_maskImage)
                controlnet_img =  Image.open(BytesIO(response.content)).convert('RGB')
                controlnet_img2 = numpy.array(controlnet_img)
                controlnet2_image['mask'] = controlnet_img2
        
        controlnet2_is_enabled = True if controlnet2_image else False
        
        controlnet3_image = None
        controlnet3_inputImage = request_data["controlnet3Image"] if 'controlnet3Image' in request_data else None
        if controlnet3_inputImage:
            if controlnet3_inputImage.startswith('data:'):
                base64_data = re.sub('^data:image/.+;base64,', '', controlnet3_inputImage)
                im_bytes = base64.b64decode(base64_data)
                image_data = BytesIO(im_bytes)
                controlnet_img = Image.open(image_data).convert('RGB')
                controlnet_img3 = numpy.array(controlnet_img)
                controlnet3_image = {'image': controlnet_img3, 'mask': numpy.zeros(controlnet_img3.shape, dtype=numpy.uint8)}
            else:
                print("downloading controlnet3Image from " + controlnet3_inputImage)
                response = requests.get(controlnet3_inputImage)
                controlnet_img =  Image.open(BytesIO(response.content)).convert('RGB')
                controlnet_img3 = numpy.array(controlnet_img)
                controlnet3_image = {'image': controlnet_img3, 'mask': numpy.zeros(controlnet_img3.shape, dtype=numpy.uint8)}
        
        controlnet3_maskImage = request_data["controlnet3MaskImage"] if 'controlnet3MaskImage' in request_data else None
        if controlnet3_maskImage and controlnet3_image:
            if controlnet3_maskImage.startswith('data:'):
                base64_data = re.sub('^data:image/.+;base64,', '', controlnet3_maskImage)
                im_bytes = base64.b64decode(base64_data)
                image_data = BytesIO(im_bytes)
                controlnet_img = Image.open(image_data).convert('RGB')
                controlnet_img3 = numpy.array(controlnet_img)
                controlnet3_image['mask'] = controlnet_img3
            else:
                print("downloading controlnet3_maskImage from " + controlnet3_maskImage)
                response = requests.get(controlnet3_maskImage)
                controlnet_img =  Image.open(BytesIO(response.content)).convert('RGB')
                controlnet_img3 = numpy.array(controlnet_img)
                controlnet3_image['mask'] = controlnet_img3
        
        controlnet3_is_enabled = True if controlnet3_image else False
        
        
        latent_couple_enabled = request_data["latentCoupleEnabled"] if 'latentCoupleEnabled' in request_data else False
        latent_couple_divisions = request_data["latentCoupleDivisions"] if 'latentCoupleDivisions' in request_data else "1:1,1:2,1:2"
        latent_couple_positions = request_data["latentCouplePositions"] if 'latentCouplePositions' in request_data else "0:0,0:0,0:1"
        latent_couple_weights = request_data["latentCoupleWeights"] if 'latentCoupleWeights' in request_data else "0.2,0.8,0.8"
        latent_couple_end_at_step = request_data["latentCoupleEndAtStep"] if 'latentCoupleEndAtStep' in request_data else 20
        composable_lora_enabled = latent_couple_enabled
        composable_lora_text_model_encoder = False
        composable_lora_diffusion_model = False

        # img2img(
        #   id_task: str,
        #   mode: int,
        #   prompt: str,
        #   negative_prompt: str,
        #   prompt_styles: Unknown,
        #   init_img: Unknown,
        #   sketch: Unknown,
        #   init_img_with_mask: Unknown,
        #   inpaint_color_sketch: Unknown,
        #   inpaint_color_sketch_orig: Unknown,
        #   init_img_inpaint: Unknown,
        #   init_mask_inpaint: Unknown,
        #   steps: int,
        #   sampler_index: int,
        #   mask_blur: int,
        #   mask_alpha: float,
        #   inpainting_fill: int,
        #   restore_faces: bool,
        #   tiling: bool,
        #   n_iter: int,
        #   batch_size: int,
        #   cfg_scale: float,
        #   image_cfg_scale: float,
        #   denoising_strength: float,
        #   seed: int,
        #   subseed: int,
        #   subseed_strength: float,
        #   seed_resize_from_h: int,
        #   seed_resize_from_w: int,
        #   seed_enable_extras: bool,
        #   height: int,
        #   width: int,
        #   resize_mode: int,
        #   inpaint_full_res: bool,
        #   inpaint_full_res_padding: int,
        #   inpainting_mask_invert: int,
        #   img2img_batch_input_dir: str,
        #   img2img_batch_output_dir: str,
        #   img2img_batch_inpaint_mask_dir: str,
        #   override_settings_texts: Unknown,
        #   *args: Unknown
        # )
        before_run(request_data)
        images, generation_info_js, stats, comments = modules.img2img.img2img(
            id_task, mode, prompt, negative_prompt, prompt_styles, init_img, sketch, 
            init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint,
            init_mask_inpaint, steps, sampler_index, mask_blur, mask_alpha, inpainting_fill, 
            restore_faces, tiling, n_iter, batch_size, cfg_scale, image_cfg_scale, denoising_strength, seed, subseed,
            subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras, height,
            width, resize_mode, inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert,
            img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
            override_settings_texts, script_args, False,
            
            controlnet1_is_enabled, 
            controlnet1_module, 
            controlnet1_model, 
            controlnet1_weight, 
            controlnet1_image,
            controlnet1_scribble_mode, 
            controlnet1_resize_mode, 
            controlnet1_rgb_to_bgr_mode, 
            controlnet1_low_vram_mode, 
            controlnet1_resolution,
            controlnet1_threshold_a, 
            controlnet1_threshold_b,
            controlnet1_guidance_start,
            controlnet1_guidance_end,
            controlnet1_guess_mode,
            
            controlnet2_is_enabled, 
            controlnet2_module, 
            controlnet2_model, 
            controlnet2_weight, 
            controlnet2_image,
            controlnet2_scribble_mode, 
            controlnet2_resize_mode, 
            controlnet2_rgb_to_bgr_mode, 
            controlnet2_low_vram_mode, 
            controlnet2_resolution,
            controlnet2_threshold_a, 
            controlnet2_threshold_b,
            controlnet2_guidance_start,
            controlnet2_guidance_end,
            controlnet2_guess_mode,
            
            
            controlnet3_is_enabled, 
            controlnet3_module, 
            controlnet3_model, 
            controlnet3_weight, 
            controlnet3_image,
            controlnet3_scribble_mode, 
            controlnet3_resize_mode, 
            controlnet3_rgb_to_bgr_mode, 
            controlnet3_low_vram_mode, 
            controlnet3_resolution,
            controlnet3_threshold_a, 
            controlnet3_threshold_b,
            controlnet3_guidance_start,
            controlnet3_guidance_end,
            controlnet3_guess_mode,            
            
            composable_lora_enabled,
            composable_lora_text_model_encoder,
            composable_lora_diffusion_model,
            
            latent_couple_enabled,
            latent_couple_divisions,
            latent_couple_positions,
            latent_couple_weights,
            latent_couple_end_at_step,
            
            )
        after_run(request_data)
                
        encoded_image = None
        second_encoded_image = None
        third_encoded_image = None
        fourth_encoded_image = None
        for image in images:
            buffered = BytesIO()
            if hasattr(image, 'save'):
                image.save(buffered, format="PNG")
            else:
                pil_image = Image.fromarray(image)
                pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            img_base64 = bytes("data:image/png;base64,", encoding='utf-8') + img_str
            
            if not encoded_image:
                encoded_image = img_base64.decode("utf-8")
            elif not second_encoded_image:
                second_encoded_image = img_base64.decode("utf-8")
            elif not third_encoded_image:
                third_encoded_image = img_base64.decode("utf-8")
            elif not fourth_encoded_image:
                fourth_encoded_image = img_base64.decode("utf-8")
            else:
                break
        if shared.state.interrupted:
            return "aborted", 500
        return {
            "generatedImage": encoded_image,
            "generatedSecondImage": second_encoded_image,
            "generatedThirdImage": third_encoded_image,
            "generatedFourthImage": fourth_encoded_image,
            "seed": str(seed),
            "stats": stats,
            "comments": comments,
        }
    except BaseException as err:
        print("error", err)
        print(traceback.format_exc())
        return "Error: {0}".format(err), 500
    finally:
        is_generating = None


@api.route('/api/upscale', methods=['POST'])
def upscale():
    global is_generating, webapi_secret
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401
    if not shared.sd_model:
        return 'still booting up', 500
    if is_generating:
        print("already generating " + is_generating)
        return 'already generating', 500
    try:
        args = request.args
        dreamId = args.get("dreamId", default="none")
        is_generating = dreamId
        request_data: Any = request.get_json()

        samplers = []
        for sampler in map(lambda x: x.name, modules.sd_samplers.samplers_for_img2img):
            samplers.append(sampler)

        init_img = None
        inputImage = request_data["inputImage"]
        if inputImage.startswith('data:'):
            base64_data = re.sub('^data:image/.+;base64,', '', inputImage)
            im_bytes = base64.b64decode(base64_data)
            image_data = BytesIO(im_bytes)
            init_img = Image.open(image_data)
        elif inputImage:
            print("downloading inputImage from " + inputImage)
            response = requests.get(inputImage)
            init_img = Image.open(BytesIO(response.content))
        else:
            return "missing input image", 500
            

        extras_mode = 0
        resize_mode = 0
        image_folder = None
        gfpgan_visibility = 0
        codeformer_visibility = 1
        codeformer_weight = 0.9
        upscaling_resize = request_data["upscaleFactor"] if "upscaleFactor" in request_data else 2
        upscaling_resize_w = 0
        upscaling_resize_h = 0
        upscaling_crop = 0
        extras_upscaler_1 = request_data["submode"] if "submode" in request_data else "None"
        extras_upscaler_2 = None
        extras_upscaler_2_visibility = 0
        input_dir = ""
        output_dir = ""
        show_extras_results = False
        upscale_first = False

        # for idx, upscaler in enumerate(shared.sd_upscalers):
        #     if upscaler.name == extras_upscaler_1_name:
        #         extras_upscaler_1 = idx
        #         break

        # print("using extras_upscaler_1", extras_upscaler_1, extras_upscaler_1_name)        

        # def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, 
        # show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, 
        # upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, 
        # extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, 
        # upscale_first: bool, save_output: bool = True):

        before_run(request_data)
        images, generation_info_js, stats = postprocessing.run_extras(
            extras_mode, resize_mode, init_img, image_folder, input_dir, output_dir, 
            show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight,
            upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop,
            extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first, save_output = False)
        after_run(request_data)
        encoded_image = None
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            img_base64 = bytes("data:image/png;base64,", encoding='utf-8') + img_str
            encoded_image = img_base64.decode("utf-8")
            break
        if shared.state.interrupted:
            return "aborted", 500
        return {
            "generatedImage": encoded_image,
            "stats": stats,
        }
    except BaseException as err:
        print("error", err)
        print(traceback.format_exc())
        return "Error: {0}".format(err), 500
    finally:
        is_generating = None
        


@api.route('/api/img2prompt', methods=['POST'])
def img2prompt():
    global is_generating, webapi_secret
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401
    if not shared.sd_model:
        return 'still booting up', 500
    if is_generating:
        print("already generating " + is_generating)
        return 'already generating', 500
    try:
        args = request.args
        dreamId = args.get("dreamId", default="none")
        is_generating = dreamId
        request_data: Any = request.get_json()

        init_img = None
        interrogator = request_data["interrogator"]
        inputImage = request_data["inputImage"]
        if inputImage.startswith('data:'):
            base64_data = re.sub('^data:image/.+;base64,', '', inputImage)
            im_bytes = base64.b64decode(base64_data)
            image_data = BytesIO(im_bytes)
            init_img = Image.open(image_data).convert('RGB')
        else:
            print("downloading inputImage from " + inputImage)
            response = requests.get(inputImage)
            init_img = Image.open(BytesIO(response.content)).convert('RGB')

        before_run(request_data)        
        if interrogator == "CLIP":
            prompt = shared.interrogator.interrogate(init_img)
        else:
            prompt = deepbooru.model.tag(init_img)
        
        after_run(request_data)
        if shared.state.interrupted:
            return "aborted", 500
        return {
            "generatedPrompt": prompt
        }
    except BaseException as err:
        print("error", err)
        print(traceback.format_exc())
        return "Error: {0}".format(err), 500
    finally:
        is_generating = None

@api.route('/assistant', methods=['POST'])
def assistant():
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401

    json:Any = request.get_json(force=True) 
    history_array = json.get('history')
    model_name = json.get('modelName')
    print("Model Name: " + model_name)
    print("\n#### HISTORY ####")
    for line in history_array:
        print(line)
    print("\n#### HISTORY ####")

    prompt = json.get('prompt')
    print("\n#### INPUT ####")
    print(prompt)
    print("\n#### INPUT ####")

    date = datetime.today().strftime('%B %d, %Y at %H:%M')
    return {
        'prompt': prompt,
        'reply': "dummy reply on " + date,
        'intermediateSteps': [],
        'openAiTokens': 0,
        'language': "en-US",
    }
    chat_agent = ChatAgent(history_array=history_array, model_name=model_name)

    try:
        with get_openai_callback() as cb:
            reply = chat_agent.agent_executor.run(input=prompt)
            print(cb.total_tokens)
            print("reply")
            print(reply)
            

    except ValueError as inst:
        print('ValueError:\n')
        print(inst)
        reply = "Sorry, there was an error processing your request."

    print("\n#### REPLY ####")
    print(reply)
    print("\n#### REPLY ####")

    pattern = r'\(([a-z]{2}-[A-Z]{2})\)'
    # Search for the local pattern in the string
    match = re.search(pattern, reply)

    language = 'en-US'  # default
    if match:
        # Get the language code
        language = match.group(1)

        # Remove the language code from the reply
        reply = re.sub(pattern, '', reply)

    print("LANG: ", language)

    sys.stdout.flush()
    return {
        'prompt': prompt,
        'reply': reply.strip(),
        'interintermediateSteps': [],
        'openAiTokens': 0,
        'language': language
    }


def webapi():
    import threading
    threading.Thread(
        target=lambda: api.run(host="0.0.0.0", port=42587, debug=True, use_reloader=False, ssl_context='adhoc')).start()


# source: https://github.com/sd-webui/stable-diffusion-webui/blob/72fb6ffe1fc76b668c822f7a2cc0934dc7bd08af/scripts/webui.py
def seed_to_int(s):
    if type(s) is int:
        return s
    if s is None or s == '':
        return random.randint(0, 2 ** 32 - 1)
    n = abs(int(s) if s.isdigit() else random.Random(s).randint(0, 2 ** 32 - 1))
    while n >= 2 ** 32:
        n = n >> 32
    return n
