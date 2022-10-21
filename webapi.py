import json, re, base64, random, requests, traceback, os, numpy
from typing import Any
from flask import Flask, json, request, jsonify
from flask_cors import CORS
from flask_sock import Sock
from PIL import Image
from io import BytesIO
from modules import shared
import modules.txt2img
import modules.img2img
import modules.extras
import modules.sd_samplers
import modules.sd_models
from pprint import pprint


# https://flask-sse.readthedocs.io/en/latest/advanced.html#channels

api = Flask(__name__)
api.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}

webapi_secret = os.getenv('WEBAPI_SECRET', None)
if webapi_secret != None:
    print("Starting with Webapi Secret: " + webapi_secret)
else:
    print("Starting without Webapi Secret")


imageResizeModes = ["Just resize", "Crop and resize", "Resize and fill"]
maskContentModes = ["fill", "original", "latent noise", "latent nothing"]
maskInpaintModes = ["Inpaint masked", "Inpaint not masked"]

sock = Sock(api)
CORS(api)
is_generating = None

def send_update(ws):
    encoded_image = ""
    if shared.state.current_image:
        buffered = BytesIO()
        shared.state.current_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        img_base64 = bytes("data:image/png;base64,", encoding='utf-8') + img_str
        encoded_image = img_base64.decode("utf-8")
        shared.state.current_image = None
            
    ws.send(json.dumps({
        "isGenerating": is_generating,
        "image": encoded_image,
        "jobNo": shared.state.job_no,
        "jobCount": shared.state.job_count,
        "samplingStep": shared.state.sampling_step,
        "samplingSteps": shared.state.sampling_steps,
    }, sort_keys=False))

@sock.route('/events')
def echo(ws):
    try:
        shared.state.register_listener(lambda: send_update(ws))
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
            shared.state.clear_listeners()
            return
        print("Exception:")
        pprint(err)
        raise err
    finally:
        shared.state.clear_listeners()
        
def after_run(request_data):
    # shared.state.clear_listeners()
    pass

def before_run(request_data):
    shared.state.sampling_step = 0
    shared.state.job_count = -1
    shared.state.job_no = 0
    shared.state.job_timestamp = shared.state.get_job_timestamp()
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
    if 'model' in request_data and request_data['model'] in models and shared.opts.sd_model_checkpoint != request_data['model']:
        print("Loading model: ", request_data['model'])
        shared.opts.sd_model_checkpoint = request_data['model']
        shared.opts.data_labels["sd_model_checkpoint"].onchange()
        dirty = True
        
    # if 'clipIgnoreLastLayers' in request_data and request_data['clipIgnoreLastLayers'] is not None and request_data['clipIgnoreLastLayers'] != shared.opts.CLIP_ignore_last_layers:
    #     shared.opts.CLIP_ignore_last_layers = request_data['clipIgnoreLastLayers']
    #     dirty = True
    if '925997e9' in shared.opts.sd_model_checkpoint:
        shared.opts.CLIP_ignore_last_layers = 2
    else:
        shared.opts.CLIP_ignore_last_layers = 0
        
    hypernetworks = [x for x in shared.hypernetworks.keys()]
    
    if 'hypernetwork' in request_data:
        if request_data['hypernetwork'] in hypernetworks and shared.opts.sd_hypernetwork != request_data['hypernetwork']:
            # print("Loading hypernetwork: ", request_data['hypernetwork'])
            shared.opts.sd_hypernetwork = request_data['hypernetwork']
            shared.opts.data_labels["sd_hypernetwork"].onchange()
            dirty = True
        elif not request_data['hypernetwork'] and shared.opts.sd_hypernetwork != "None":
            print("Unloading hypernetwork")
            shared.opts.sd_hypernetwork = "None"
            shared.opts.data_labels["sd_hypernetwork"].onchange()
            dirty = True
            
    if 'faceRestoreModel' in request_data and request_data['faceRestoreModel'] is not None:
        if request_data['faceRestoreModel'] != shared.opts.face_restoration_model:
            shared.opts.face_restoration_model = request_data['faceRestoreModel']
    elif shared.opts.face_restoration_model is not None and shared.opts.face_restoration_model != "None":
        shared.opts.face_restoration_model = "None"
            
    if 'faceRestoreStrength' in request_data and request_data['faceRestoreStrength'] is not None:
        if request_data['faceRestoreStrength'] != shared.opts.code_former_weight:
            shared.opts.code_former_weight = request_data['faceRestoreStrength']
    elif shared.opts.code_former_weight > 0:
        shared.opts.face_restoration_model = 0
            
    if dirty:
        shared.opts.save(shared.config_filename)

@api.route('/api/endpoints', methods=['GET'])
def list_endpoints():
    global webapi_secret
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        print("got wrong secret: " + request.headers.get('webapi-secret', "should not happen"))
        return 'wrong secret', 401
    
    imageSize = {
        "min": 256,
        "max": 2048,
        "step": 64,
        "default": 512
    }
    
    samplers = {}
    for sampler in map(lambda x: x.name, modules.sd_samplers.samplers):
        samplers[sampler] = sampler
        
    samplers_img2img = {}
    for sampler in map(lambda x: x.name, modules.sd_samplers.samplers_for_img2img):
        samplers_img2img[sampler] = sampler        
        
    upscaleFactor = {
        "min": 1, 
        "max": 4, 
        "step": 0.5, 
        "default": 2
    }
    
    models = modules.sd_models.checkpoint_tiles()
    model = shared.opts.sd_model_checkpoint
    
    hypernetworks = [x for x in shared.hypernetworks.keys()]
    hypernetwork = shared.opts.sd_hypernetwork
    if hypernetwork == "None":
        hypernetwork = None
        
    # clipIgnoreLastLayers = {
    #             "min": 0,
    #             "max": 5,
    #             "step": 1,
    #             "default": shared.opts.CLIP_ignore_last_layers
    # }
    
    faceRestoreModels = [x.name() for x in shared.face_restorers if x.name() != "None"]
    faceRestoreStrength = {
        "min": 0,
        "max": 1,
        "step": 0.05,
        "default": 0.5
    }
    
    return jsonify([
        {
            "name": "Text to Image",
            "mode": "txt2img",
            "path": "/api/txt2img",
            "inputs": {
                "model": model,
                "models": models,
                "hypernetwork": hypernetwork,
                "hypernetworks": hypernetworks,
                
                "isHighresFix":True,
                "isHighresFixScaleLatent":True,
                "isTiling":True,
                
                "faceRestoreModels": faceRestoreModels,
                "faceRestoreStrength": faceRestoreStrength,
                
                "prompt": True,
                "negativePrompt": True,
                "sampleSteps": {
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "default": 30
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
                "resizeSeedFromWidth": {
                    "min": 0,
                    "max": 2048,
                    "step": 64,
                    "default": 0
                },
                "resizeSeedFromHeight": {
                    "min": 0,
                    "max": 2048,
                    "step": 64,
                    "default": 0
                },
                "firstPassWidth": {
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "default": 0
                },
                "firstPassHeight": {
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "default": 0
                },
                "seed": True,
                "variationSeed": True,
                "variationStrength": {
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "default": 0
                },
                "samplers": samplers,
                "sampler": "Euler",
                "imageWidth": imageSize,
                "imageHeight": imageSize,
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
                "model": model,
                "models": models,
                "hypernetwork": hypernetwork,
                "hypernetworks": hypernetworks,
                "image": True,
                "mask": True,
                "prompt": True,
                "negativePrompt": True,
                
                "imageResizeMode": "Just resize",
                "imageResizeModes": imageResizeModes,
                "maskContentMode": "fill",
                "maskContentModes": maskContentModes,
                "maskInpaintMode": "Inpaint masked",
                "maskInpaintModes": maskInpaintModes,
                
                "maskBlurStrength": {
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "default": 4
                },
                "maskInpaintFullResolution": True,
                "maskInpaintFullResolutionPadding": {
                    "min": 0,
                    "max": 256,
                    "step": 4,
                    "default": 32
                },
                
                "sampleSteps": {
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "default": 30
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
                "resizeSeedFromWidth": {
                    "min": 0,
                    "max": 2048,
                    "step": 64,
                    "default": 0
                },
                "resizeSeedFromHeight": {
                    "min": 0,
                    "max": 2048,
                    "step": 64,
                    "default": 0
                },
                "isTiling":True,
                "seed": True,
                "variationSeed": True,
                "variationStrength": {
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "default": 0
                },
                "faceRestoreModels": faceRestoreModels,
                "faceRestoreStrength": faceRestoreStrength,
                "samplers": samplers_img2img,
                "sampler": "DDIM",
                "imageWidth": imageSize,
                "imageHeight": imageSize,
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
                    "name": "No Upscale",
                    "submode": "None",
                    "inputs": {
                    }
                },
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
                "faceRestoreModels": faceRestoreModels,
                "faceRestoreStrength": faceRestoreStrength
            },
            "outputs": {
                "image": "png"
            }
        }
    ]), 200
    

@api.route('/api/txt2img', methods=['POST'])
def txt2img():
    global is_generating, webapi_secret
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401
    if not shared.sd_model:
        return 'still booting up', 500
    if is_generating:
        return 'already generating', 500
    try:
        args = request.args        
        request_data: Any = request.get_json()
        # print(request_data)
        
        dreamId = args.get("dreamId", default="none")
        is_generating = dreamId
        
        samplers = []
        for sampler in map(lambda x: x.name, modules.sd_samplers.samplers):
            samplers.append(sampler)
        
        prompt = request_data["prompt"] if "prompt" in request_data else "???"
        negative_prompt = request_data["negativePrompt"]  if "negativePrompt" in request_data else ""
        prompt_style = ""
        prompt_style2 = ""
        steps = request_data["sampleSteps"] if "sampleSteps" in request_data else 10
        sampler = request_data["sampler"] if "sampler" in request_data else "LMS"
        sampler_index = samplers.index(sampler) if sampler in samplers else 0
        restore_faces = request_data["faceRestoreStrength"] > 0 if "faceRestoreStrength" in request_data else False
        tiling = request_data["isTiling"] if "isTiling" in request_data else False
        n_iter = 1
        batch_size = 1
        cfg_scale = request_data["guidanceScale"] if "guidanceScale" in request_data else 7.5
        seed = seed_to_int(request_data["seed"] if "seed" in request_data else "")
        subseed = request_data["variationSeed"] if "variationSeed" in request_data else 0
        subseed_strength = request_data["variationStrength"] if "variationStrength" in request_data else 0
        seed_resize_from_w = request_data["resizeSeedFromWidth"] if "resizeSeedFromWidth" in request_data else 0
        seed_resize_from_h = request_data["resizeSeedFromHeight"] if "resizeSeedFromHeight" in request_data else 0
        seed_enable_extras = seed_resize_from_w > 0 or seed_resize_from_h > 0 or subseed_strength > 0
        height = request_data["height"] if "height" in request_data else 512
        width = request_data["width"] if "width" in request_data else 512
        enable_hr = request_data["isHighresFix"] if "isHighresFix" in request_data else False
        denoising_strength = request_data["denoisingStrength"] if "denoisingStrength" in request_data else 0.75
        script_args = 0 # no script
        firstphase_width = request_data["firstPassWidth"] if "firstPassWidth" in request_data else 0
        firstphase_height = request_data["firstPassHeight"] if "firstPassHeight" in request_data else 0
        
        # txt2img(prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, steps: int, sampler_index: int, 
        # restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, 
        # subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, 
        # height: int, width: int, enable_hr: bool, denoising_strength: float, 
        # firstphase_width: int, firstphase_height: int, *args
        
        before_run(request_data)
        images, generation_info_js, stats = modules.txt2img.txt2img(prompt, negative_prompt, prompt_style, prompt_style2, steps, 
                                sampler_index, restore_faces, tiling, n_iter, batch_size, 
                                cfg_scale, seed, subseed, subseed_strength, seed_resize_from_h,
                                seed_resize_from_w, seed_enable_extras, height, width, enable_hr, 
                                denoising_strength, firstphase_width, firstphase_height, script_args)
        after_run(request_data)
        is_generating = None
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
            "seed": str(seed),
            "stats": stats,
        }
    except BaseException as err:
        print("error", err)
        print(traceback.format_exc())
        is_generating = None
        return "Error: {0}".format(err), 500
    

@api.route('/api/img2img', methods=['POST'])
def img2img():
    global is_generating, webapi_secret
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401
    if is_generating:
        return 'already generating', 500
    if not shared.sd_model:
        return 'still booting up', 500
    try:
        args = request.args        
        request_data: Any = request.get_json()
        # print(request_data)
        
        dreamId = args.get("dreamId", default="none")
        is_generating = dreamId
        
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
        else:
            print("downloading inputImage from " + inputImage)
            response = requests.get(inputImage)
            init_img = Image.open(BytesIO(response.content))
        
        
        mode = 0
        mask_mode = 0
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
                    
                   
                init_img_with_mask = {"image": init_img.convert("RGBA"), "mask": mask_info.resize(init_img.size)}
                init_img = None
                mode = 1
                mask_mode = 0

        prompt = request_data["prompt"] if "prompt" in request_data else ""
        negative_prompt = request_data["negativePrompt"]  if "negativePrompt" in request_data else ""
        prompt_style = ""
        prompt_style2 = ""
        #init_img = None
        #init_img_with_mask = None
        init_img_inpaint = None
        init_mask_inpaint = None
        try:
            resize_mode = imageResizeModes.index(request_data["imageResizeMode"]) if "imageResizeMode" in request_data else 0
        except ValueError:
            resize_mode = 0
        steps = request_data["sampleSteps"] if "sampleSteps" in request_data else 10
        sampler = request_data["sampler"] if "sampler" in request_data else "LMS"
        sampler_index = samplers.index(sampler) if sampler in samplers else 0
        mask_blur = request_data["maskBlurStrength"] if "maskBlurStrength" in request_data else 4
        try:
            inpainting_fill = maskContentModes.index(request_data["maskContentMode"]) if "maskContentMode" in request_data else 0
        except ValueError:
            inpainting_fill = 0
        restore_faces = request_data["faceRestoreStrength"] > 0 if "faceRestoreStrength" in request_data else False
        tiling = request_data["isTiling"] if "isTiling" in request_data else False
        n_iter = 1
        inpaint_full_res = request_data["maskInpaintFullResolution"] if "maskInpaintFullResolution" in request_data else False
        inpaint_full_res_padding = request_data["maskInpaintFullResolutionPadding"] if "maskInpaintFullResolutionPadding" in request_data else 0
        try:
            inpainting_mask_invert = maskInpaintModes.index(request_data["maskInpaintMode"]) if "maskInpaintMode" in request_data else 0
        except ValueError:
            inpainting_mask_invert = 0
        batch_size = 1
        cfg_scale = request_data["guidanceScale"] if "guidanceScale" in request_data else 7.5
        seed = seed_to_int(request_data["seed"] if "seed" in request_data else "")
        subseed = request_data["variationSeed"] if "variationSeed" in request_data else 0
        subseed_strength = request_data["variationStrength"] if "variationStrength" in request_data else 0
        seed_resize_from_w = request_data["resizeSeedFromWidth"] if "resizeSeedFromWidth" in request_data else 0
        seed_resize_from_h = request_data["resizeSeedFromHeight"] if "resizeSeedFromHeight" in request_data else 0
        seed_enable_extras = seed_resize_from_w > 0 or seed_resize_from_h > 0 or subseed_strength > 0
        img2img_batch_input_dir = ""
        img2img_batch_output_dir = ""
        height = request_data["height"] if "height" in request_data else 512
        width = request_data["width"] if "width" in request_data else 512
        denoising_strength = request_data["denoisingStrength"] if "denoisingStrength" in request_data else 0.75
        script_args = 0
        
        
        
        # img2img(mode: int, prompt: str, negative_prompt: str, prompt_style: str, 
        # prompt_style2: str, init_img, init_img_with_mask, init_img_inpaint, init_mask_inpaint, 
        # mask_mode, steps: int, sampler_index: int, mask_blur: int, inpainting_fill: int, 
        # restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, 
        # denoising_strength: float, seed: int, subseed: int, subseed_strength: float, 
        # seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, 
        # height: int, width: int, resize_mode: int, inpaint_full_res: bool, 
        # inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, 
        # img2img_batch_output_dir: str, *args
        before_run(request_data)
        images, generation_info_js, stats = modules.img2img.img2img(mode, prompt, negative_prompt, prompt_style, prompt_style2, init_img, 
                                init_img_with_mask, init_img_inpaint, init_mask_inpaint, mask_mode, steps, 
                                sampler_index, mask_blur, inpainting_fill, restore_faces, tiling, n_iter, batch_size, 
                                cfg_scale, denoising_strength, seed, subseed, subseed_strength, seed_resize_from_h,
                                seed_resize_from_w, seed_enable_extras, height, width, resize_mode, inpaint_full_res, 
                                inpaint_full_res_padding,  inpainting_mask_invert, img2img_batch_input_dir,
                                img2img_batch_output_dir, script_args)
        after_run(request_data)
        is_generating = None
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
            "seed": str(seed),
            "stats": stats,
        }
    except BaseException as err:
        print("error", err)
        print(traceback.format_exc())
        is_generating = None
        return "Error: {0}".format(err), 500
    

@api.route('/api/upscale', methods=['POST'])
def upscale():
    global is_generating, webapi_secret
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401
    if is_generating:
        return 'already generating', 500
    if not shared.sd_model:
        return 'still booting up', 500
    try:
        args = request.args        
        request_data: Any = request.get_json()
        # print(request_data)
        
        dreamId = args.get("dreamId", default="none")
        is_generating = dreamId
        
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
        else:
            print("downloading inputImage from " + inputImage)
            response = requests.get(inputImage)
            init_img = Image.open(BytesIO(response.content))
        
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
        extras_upscaler_1_name = request_data["submode"] if "submode" in request_data else "None"
        extras_upscaler_1 = 0
        extras_upscaler_2 = 0
        extras_upscaler_2_visibility = 0
        input_dir = ""
        output_dir = ""
        show_extras_results = False
        
        for idx, upscaler in enumerate(shared.sd_upscalers):
            if upscaler.name == extras_upscaler_1_name:
                extras_upscaler_1 = idx
                break
        
        # print("using extras_upscaler_1", extras_upscaler_1, extras_upscaler_1_name)        
                
        # run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, 
        # show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, 
        # upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, 
        # extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility
        
        before_run(request_data)
        images, generation_info_js, stats = modules.extras.run_extras(extras_mode, resize_mode, init_img, 
                        image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, 
                        upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, 
                        extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility)
        after_run(request_data)
        is_generating = None
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
        is_generating = None
        return "Error: {0}".format(err), 500
    
        
@api.route('/api/img2prompt', methods=['POST'])
def img2prompt():
    global is_generating, webapi_secret
    if webapi_secret and request.headers.get('webapi-secret', None) != webapi_secret:
        return 'wrong secret', 401
    if is_generating:
        return 'already generating', 500
    if not shared.sd_model:
        return 'still booting up', 500
    try:
        args = request.args        
        request_data: Any = request.get_json()
        # print(request_data)
        
        dreamId = args.get("dreamId", default="none")
        is_generating = dreamId
        
        init_img = None
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
        prompt = shared.interrogator.interrogate(init_img)
        after_run(request_data)
        is_generating = None
        if shared.state.interrupted:
            return "aborted", 500
        return {
            "generatedPrompt": prompt
        }
    except BaseException as err:
        print("error", err)
        print(traceback.format_exc())
        is_generating = None
        return "Error: {0}".format(err), 500
    
        
def webapi():
    import threading
    threading.Thread(target=lambda: api.run(host="0.0.0.0", port=42587, debug=True, use_reloader=False, ssl_context='adhoc')).start()

# source: https://github.com/sd-webui/stable-diffusion-webui/blob/72fb6ffe1fc76b668c822f7a2cc0934dc7bd08af/scripts/webui.py
def seed_to_int(s):
    if type(s) is int:
        return s
    if s is None or s == '':
        return random.randint(0, 2**32 - 1)
    n = abs(int(s) if s.isdigit() else random.Random(s).randint(0, 2**32 - 1))
    while n >= 2**32:
        n = n >> 32
    return n
