import json, re, base64, random, requests
from flask import Flask, json, request
from flask_cors import CORS
from flask_sock import Sock
from PIL import Image
from io import BytesIO
from modules import shared
import modules.txt2img
import modules.img2img
import modules.extras
import modules.sd_samplers

# https://flask-sse.readthedocs.io/en/latest/advanced.html#channels

api = Flask(__name__)
api.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
api.config['SECRET_KEY'] = 'QzBFr3yuPeSBaEngJyagGPqRakIhjDSz'

sock = Sock(api)
CORS(api)
is_generating = None

def send_update(ws):
    ws.send(json.dumps({
        "isGenerating": is_generating,
        "interrupted": shared.state.interrupted,
        "job": shared.state.job,
        "jobNo": shared.state.job_no,
        "jobCount": shared.state.job_count,
        "samplingStep": shared.state.sampling_step,
        "samplingSteps": shared.state.sampling_steps,
    }, sort_keys=True, indent=4))

@sock.route('/events')


def echo(ws):
    try:
        while True:
            shared.state.register_listener(lambda: send_update(ws))
            data = ws.receive()
            if data == 'close':
                break
            send_update(ws)
    finally:
        shared.state.clear_listeners()

@api.route('/api/endpoints', methods=['GET'])
def list_endpoints():
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
    
    return [
    {
        "name": "Text to Image",
        "mode": "txt2img",
        "path": "/api/txt2img",
        "inputs": {
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
    

@api.route('/api/txt2img', methods=['POST'])
def txt2img():
    global is_generating
    if not shared.sd_model:
        return 'still booting up', 500
    if is_generating:
        return 'already generating', 500
    try:
        args = request.args        
        request_data: Any = request.get_json()
        print(request_data)
        
        dreamId = args.get("dreamId", default="none")
        is_generating = dreamId
        
        samplers = []
        for sampler in map(lambda x: x.name, modules.sd_samplers.samplers):
            samplers.append(sampler)
        
        prompt = request_data["prompt"] if "prompt" in request_data else ""
        negative_prompt = request_data["negativePrompt"]  if "negativePrompt" in request_data else ""
        prompt_style = ""
        prompt_style2 = ""
        steps = request_data["sampleSteps"] if "sampleSteps" in request_data else 10
        sampler = request_data["sampler"] if "sampler" in request_data else "LMS"
        sampler_index = samplers.index(sampler) if sampler in samplers else 0
        restore_faces = False
        tiling = False
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
        enable_hr = False
        scale_latent = False
        denoising_strength = request_data["denoisingStrength"] if "denoisingStrength" in request_data else 0.75
        script_args = 0
        
        # txt2img(prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, steps: int, 
        # sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, 
        # cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, 
        # seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, enable_hr: bool, 
        # scale_latent: bool, denoising_strength: float, *args):
        images, generation_info_js, stats = modules.txt2img.txt2img(prompt, negative_prompt, prompt_style, prompt_style2, steps, 
                                sampler_index, restore_faces, tiling, n_iter, batch_size, 
                                cfg_scale, seed, subseed, subseed_strength, seed_resize_from_h,
                                seed_resize_from_w, seed_enable_extras, height, width, enable_hr, 
                                scale_latent, denoising_strength, script_args)
        is_generating = None
        encoded_images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            img_base64 = bytes("data:image/png;base64,", encoding='utf-8') + img_str
            encoded_images.append(img_base64.decode("utf-8"))
        return {
            "generatedImage": encoded_images[0],
            "seed": str(seed),
            "stats": stats,
        }
    except BaseException as err:
        print("error", err)
        is_generating = None
        return "Error: {0}".format(err), 500
    

@api.route('/api/img2img', methods=['POST'])
def img2img():
    global is_generating
    if is_generating:
        return 'already generating', 500
    if not shared.sd_model:
        return 'still booting up', 500
    try:
        args = request.args        
        request_data: Any = request.get_json()
        print(request_data)
        
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
                init_img = None
                init_img_with_mask = {"image": init_img, "mask": mask_info}
                mode = 1

        prompt = request_data["prompt"] if "prompt" in request_data else ""
        negative_prompt = request_data["negativePrompt"]  if "negativePrompt" in request_data else ""
        prompt_style = ""
        prompt_style2 = ""
        #init_img = None
        #init_img_with_mask = None
        init_img_inpaint = None
        init_mask_inpaint = None
        mask_mode = 0
        resize_mode = 0
        steps = request_data["sampleSteps"] if "sampleSteps" in request_data else 10
        sampler = request_data["sampler"] if "sampler" in request_data else "LMS"
        sampler_index = samplers.index(sampler) if sampler in samplers else 0
        mask_blur = 0
        inpainting_fill = 0
        restore_faces = False
        tiling = False
        n_iter = 1
        inpaint_full_res = False
        inpaint_full_res_padding = 0
        inpainting_mask_invert = 0
        batch_size = 1
        cfg_scale = request_data["guidanceScale"] if "guidanceScale" in request_data else 7.5
        seed = seed_to_int(request_data["seed"] if "seed" in request_data else "")
        subseed = 0
        subseed_strength = 0
        seed_resize_from_h = 0
        seed_resize_from_w = 0
        seed_enable_extras = False 
        img2img_batch_input_dir = ""
        img2img_batch_output_dir = ""
        height = request_data["height"] if "height" in request_data else 512
        width = request_data["width"] if "width" in request_data else 512
        enable_hr = False
        scale_latent = False
        denoising_strength = request_data["denoisingStrength"] if "denoisingStrength" in request_data else 0.75
        script_args = 0
        
        
        
        # mode: int, prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, init_img, 
        # init_img_with_mask, init_img_inpaint, init_mask_inpaint, mask_mode, steps: int, 
        # sampler_index: int, mask_blur: int, inpainting_fill: int, restore_faces: bool, tiling: bool, 
        # n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, seed: int, 
        # subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, 
        # seed_enable_extras: bool, height: int, width: int, resize_mode: int, inpaint_full_res: bool,
        # inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, 
        # img2img_batch_output_dir: str, *args
        images, generation_info_js, stats = modules.img2img.img2img(mode, prompt, negative_prompt, prompt_style, prompt_style2, init_img, 
                                init_img_with_mask, init_img_inpaint, init_mask_inpaint, mask_mode, steps, 
                                sampler_index, mask_blur, inpainting_fill, restore_faces, tiling, n_iter, batch_size, 
                                cfg_scale, denoising_strength, seed, subseed, subseed_strength, seed_resize_from_h,
                                seed_resize_from_w, seed_enable_extras, height, width, resize_mode, inpaint_full_res, 
                                inpaint_full_res_padding,  inpainting_mask_invert, img2img_batch_input_dir,
                                img2img_batch_output_dir, scale_latent, script_args)
        is_generating = None
        encoded_images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            img_base64 = bytes("data:image/png;base64,", encoding='utf-8') + img_str
            encoded_images.append(img_base64.decode("utf-8"))
        return {
            "generatedImage": encoded_images[0],
            "seed": str(seed),
            "stats": stats,
        }
    except BaseException as err:
        print("error", err)
        is_generating = None
        return "Error: {0}".format(err), 500
    

@api.route('/api/upscale', methods=['POST'])
def upscale():
    global is_generating
    if is_generating:
        return 'already generating', 500
    if not shared.sd_model:
        return 'still booting up', 500
    try:
        args = request.args        
        request_data: Any = request.get_json()
        print(request_data)
        
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
        image_folder = None
        gfpgan_visibility = 0 
        codeformer_visibility = 1
        codeformer_weight = 0.9
        upscaling_resize = request_data["upscaleFactor"] if "upscaleFactor" in request_data else 2
        extras_upscaler_1_name = request_data["submode"] if "submode" in request_data else "None"
        extras_upscaler_1 = 0
        extras_upscaler_2 = 0
        extras_upscaler_2_visibility = 0
        
        for idx, upscaler in enumerate(shared.sd_upscalers):
            if upscaler.name == extras_upscaler_1_name:
                extras_upscaler_1 = idx
                break
        
        print("using extras_upscaler_1", extras_upscaler_1, extras_upscaler_1_name)        
        
        # extras_mode, image, image_folder, gfpgan_visibility, codeformer_visibility, 
        # codeformer_weight, upscaling_resize, extras_upscaler_1, extras_upscaler_2, 
        # extras_upscaler_2_visibility
        
        images, generation_info_js, stats = modules.extras.run_extras(extras_mode, init_img, 
                        image_folder, gfpgan_visibility, codeformer_visibility, codeformer_weight, 
                        upscaling_resize, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility)
        is_generating = None
        encoded_images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue())
            img_base64 = bytes("data:image/png;base64,", encoding='utf-8') + img_str
            encoded_images.append(img_base64.decode("utf-8"))
        return {
            "generatedImage": encoded_images[0],
            "stats": stats,
        }
    except BaseException as err:
        print("error", err)
        is_generating = None
        return "Error: {0}".format(err), 500
    
        
@api.route('/api/img2prompt', methods=['POST'])
def img2prompt():
    global is_generating
    if is_generating:
        return 'already generating', 500
    if not shared.sd_model:
        return 'still booting up', 500
    try:
        args = request.args        
        request_data: Any = request.get_json()
        print(request_data)
        
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
        
        prompt = shared.interrogator.interrogate(init_img)
        is_generating = None
        return {
            "generatedPrompt": prompt
        }
    except BaseException as err:
        print("error", err)
        is_generating = None
        return "Error: {0}".format(err), 500
    
        
def webapi():
    import threading
    threading.Thread(target=lambda: api.run(host="0.0.0.0", port=42587, debug=True, use_reloader=False, ssl_context='adhoc')).start()

def seed_to_int(s):
    if type(s) is int:
        return s
    if s is None or s == '':
        return random.randint(0, 2**32 - 1)
    n = abs(int(s) if s.isdigit() else random.Random(s).randint(0, 2**32 - 1))
    while n >= 2**32:
        n = n >> 32
    return n
