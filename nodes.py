import comfy.samplers
import folder_paths 
import os
import torch
import torch.nn.functional as F
import comfy
from pathlib import Path


def get_sampler_list():
    return ["none"] + comfy.samplers.KSampler.SAMPLERS

def get_scheduler_list():
    return ["none"] + comfy.samplers.KSampler.SCHEDULERS

def get_model_list():
    checkpoints = folder_paths.get_filename_list("checkpoints")
    return ["none"] + checkpoints

def get_diffusion_model_file_list():
    diffusion_models = folder_paths.get_filename_list("diffusion_models")
    return ["none"] + diffusion_models
    
def get_checkpoint_list():
    """Returns a list of checkpoint files plus 'none'."""
    checkpoints = folder_paths.get_filename_list("checkpoints")
    return ["none"] + checkpoints

def get_vae_list():
    """Returns a list of VAE files plus 'none'."""
    vae_files = folder_paths.get_filename_list("vae")   # key "vae"
    return ["none"] + vae_files
    
def get_text_encoder_list():
    return ["none"] + folder_paths.get_filename_list("text_encoders")

def get_lora_list():
    """
    List of all LORA files from the `loras/` folder.
    The full path will be used as the value in the dropdown menu.
    """
    loras = folder_paths.get_filename_list("loras")      # <-- full path
    return ["none"] + loras
    
class SamplerGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        samplers = get_sampler_list()
        inputs = {"required": {f"sampler_{i+1}": (samplers, {"default": "none"}) for i in range(10)}}
        return inputs
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"sampler_{i+1}")
            if name and name != "none": selected.append(name)
        return (", ".join(selected),)

class SchedulerGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        schedulers = get_scheduler_list()
        inputs = {"required": {f"scheduler_{i+1}": (schedulers, {"default": "none"}) for i in range(10)}}
        return inputs
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"scheduler_{i+1}")
            if name and name != "none": selected.append(name)
        return (", ".join(selected),)

class ModelGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list()
        inputs = {"required": {f"model_{i+1}": (models, {"default": "none"}) for i in range(10)}}
        return inputs
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("MODEL_STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"model_{i+1}")
            if name and name != "none": selected.append(name)
        return (", ".join(selected),)

class DiffusionModelGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_diffusion_model_file_list()
        inputs = {"required": {f"diff_model_{i+1}": (models, {"default": "none"}) for i in range(10)}} 
        return inputs
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DIFF_MODEL_STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"
    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"diff_model_{i+1}")
            if name and name != "none": selected.append(name)
        return (", ".join(selected),)

class AnyAdapterNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Switched from "required" to "optional" – this node can accept any type.
        return {"optional": {"input_any": ("*",)}}

    RETURN_TYPES = ("*",)
    FUNCTION = "adapt"
    CATEGORY = "utils"

    def adapt(self, input_any=None):
        """
        If the input is not connected, `input_any` will be None.
        In that case we simply return nothing (or None) so that the node
        passes an empty result downstream.
        """
        if input_any is None:
            # Returning a single element – ComfyUI expects at least one output value.
            return (None,)
        return (input_any,)

class CheckpointSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Input: a single checkpoint. The combo‑menu will be built automatically.
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    # First element in the tuple → combo, second → STRING
    RETURN_TYPES = (
        folder_paths.get_filename_list("checkpoints"),
        "STRING",
    )
    RETURN_NAMES = ("ckpt_name", "ckpt_name_str")
    CATEGORY = "utils"   # Folder in UI

    FUNCTION = "get_ckpt_name"

    def get_ckpt_name(self, ckpt_name):
        return ckpt_name, ckpt_name

class DiffusionModelSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Input: one file from the diffusion_models folder (combo‑menu)
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                ),
            }
        }

    # First element → combo, second → STRING
    RETURN_TYPES = (
        folder_paths.get_filename_list("diffusion_models"),
        "STRING",
    )
    RETURN_NAMES = ("model_name", "model_name_str")
    CATEGORY = "utils"     # Folder in UI

    FUNCTION = "get_model"

    def get_model(self, model_name):
        return model_name, model_name

class VAEGeneratorNode:
    """
    Generator of a list of VAE files.
    Parameters: 10 dropdowns → string of the selected values.
    """
    @classmethod
    def INPUT_TYPES(cls):
        vaes = get_vae_list()
        inputs = {
            "required": {f"vae_{i+1}": (vaes, {"default": "none"}) for i in range(10)}
        }
        return inputs

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"

    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"vae_{i+1}")
            if name and name != "none":
                selected.append(name)
        return (", ".join(selected),)  

class TextEncoderGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        encoders = get_text_encoder_list()
        inputs = {
            "required": {f"text_enc_{i+1}": (encoders, {"default": "none"}) for i in range(10)}
        }
        return inputs

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"

    def generate_string(self, **kwargs):
        selected = []
        for i in range(10):
            name = kwargs.get(f"text_enc_{i+1}")
            if name and name != "none":
                selected.append(name)
        return (", ".join(selected),)

class VAESelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Input: one VAE. Combo‑menu will be built from files in the “vae” folder.
                "vae_name": (
                    folder_paths.get_filename_list("vae"),
                ),
            }
        }

    # First output – combo (can be connected to Load VAE, etc.)
    # Second – string with the same name
    RETURN_TYPES = (
        folder_paths.get_filename_list("vae"),  # combo‑output
        "STRING",                               # plain text output
    )
    RETURN_NAMES = ("vae_name", "vae_name_str")
    CATEGORY = "utils"     # subfolder in UI

    FUNCTION = "get_vae"

    def get_vae(self, vae_name):
        return vae_name, vae_name

class TextEncoderSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Input: one Text‑Encoder. Combo‑menu will be built from files in the “text_encoders” folder.
                "enc_name": (
                    folder_paths.get_filename_list("text_encoders"),
                ),
            }
        }

    RETURN_TYPES = (
        folder_paths.get_filename_list("text_encoders"),  # combo‑output
        "STRING",                                        # plain text output
    )
    RETURN_NAMES = ("enc_name", "enc_name_str")
    CATEGORY = "utils"

    FUNCTION = "get_encoder"

    def get_encoder(self, enc_name):
        return enc_name, enc_name

class StringToIntNode:
    """
    Accepts a string and attempts to convert it into an integer.
    If conversion fails – returns 0 (or you could raise an error instead).
    """
    @classmethod
    def INPUT_TYPES(cls):
        # "STRING" guarantees that the user sees a text input field
        return {"required": {"text_value": ("STRING",)}}

    RETURN_TYPES = ("INT",)            # output – integer
    FUNCTION = "convert"
    CATEGORY = "utils"

    def convert(self, text_value):
        try:
            return (int(text_value),)
        except Exception as e:
            # Log the error in the console; here we simply return 0
            print(f"[StringToIntNode] Conversion error for '{text_value}': {e}")
            return (0,)

class StringToFloatNode:
    """
    Accepts a string and attempts to convert it into a floating‑point number.
    If conversion fails – returns 0.0.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text_value": ("STRING",)}}

    RETURN_TYPES = ("FLOAT",)          # output – float
    FUNCTION = "convert"
    CATEGORY = "utils"

    def convert(self, text_value):
        try:
            return (float(text_value),)
        except Exception as e:
            print(f"[StringToFloatNode] Conversion error for '{text_value}': {e}")
            return (0.0,)

class TextConcatNode:
       # modified node TextConcat from https://github.com/bash-j/mikey_nodes
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"delimiter": ("STRING", {"default": " "})},
            "optional": {f"text{i}": ("STRING", {"default": ""})
                         for i in range(1, 6)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concat"
    CATEGORY = "utils"

    def concat(self,
               delimiter,
               text1="", text2="", text3="",
               text4="", text5=""):
        """Collect non‑empty strings into a single text."""
        texts = [t for t in (text1, text2, text3, text4, text5) if t]
        return (delimiter.join(texts),)     

class LORASelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        loras = get_lora_list()
        inputs = {"required": {}}
        for i in range(10):
            # Dropdown for the full path of a LoRA file
            inputs["required"][f"lora_{i+1}"] = (loras, {"default": "none"})
            # Corresponding weight input
            inputs["required"][f"weight_{i+1}"] = (
                "FLOAT",
                {"default": 1.0, "min": 0.00, "max": 10.00, "step": 0.01},
            )
        return inputs

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_string"
    CATEGORY = "utils"

    def generate_string(self, **kwargs):
        parts = []
        for i in range(10):
            # Full path from the dropdown
            path = kwargs.get(f"lora_{i+1}")
            weight = kwargs.get(f"weight_{i+1}")

            if not path or path == "none":
                continue

            # Take only the file name without extension
            name_without_ext = os.path.splitext(os.path.basename(path))[0]
            weight_str = f"{float(weight):.2f}"
            parts.append(f"<lora:{name_without_ext}:{weight_str}>")

        return (", ".join(parts),)

class ClipSkipSliderNode:
    """
    Emits an integer in the range [-24 … -1].
    The slider is bounded by its min/max attributes – the user cannot
    select a value outside this interval.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 1. Input name → type INT
                # 2. Property dictionary → default, min, max
                "value": ("INT", {"default": -1, "min": -24, "max": -1})
            }
        }

    RETURN_TYPES = ("INT",)          # single output – integer
    RETURN_NAMES = ("value",)        # optional – gives the output a name
    FUNCTION = "get_value"           # method that will be invoked
    CATEGORY = "utils"               # subfolder in UI

    def get_value(self, value):
        """
        Receives the slider's current integer value and returns it.
        The return is wrapped in a tuple because the node interface expects
        an iterable of outputs.
        """
        return (value,)

class PonyPrefixesNode:
    
    
    """
    
    Score     – 5 variants
        "-"               → None
        "Everything"      → "score_9, score_8_up, score_7_up, score_6_up, score_5_up, "
        "Average"         → "score_9, score_8_up, score_7_up, score_6_up, score_5_up, "
        "Good"            → "score_9, score_8_up, score_7_up, "
        "Only the best"   → "score_9, "

    Rating    – 4 variants
        "-"               → None
        "Safe"            → "rating_safe, "
        "Questionable"    → "rating_questionable, "
        "Explicit"        → "rating_explicit, "

    Source    – 5 variants
        "-"               → None
        "Anime"           → "source_anime, "
        "Furry"           → "source_furry, "
        "Cartoon"         → "source_cartoon, "
        "Pony"            → "source_pony, "

   In results combined string (order: Score → Rating → Source).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 1 list – Score
                "score": (
                    [
                        "-",
                        "Everything",
                        "Average",
                        "Good",
                        "Only the best",
                    ],
                ),
                # 2 list – Rating
                "rating": (
                    [
                        "-",
                        "Safe",
                        "Questionable",
                        "Explicit",
                    ],
                ),
                # 3 list – Source
                "source": (
                    [
                        "-",
                        "Anime",
                        "Furry",
                        "Cartoon",
                        "Pony",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_string",)   # output name
    FUNCTION = "generate"
    CATEGORY = "utils"

    # mappings
    _SCORE_MAP = {
        "Everything":     "score_9, score_8_up, score_7_up, score_6_up, score_5_up, ",
        "Average":        "score_9, score_8_up, score_7_up, score_6_up, score_5_up, ",
        "Good":           "score_9, score_8_up, score_7_up, ",
        "Only the best":  "score_9, ",
    }

    _RATING_MAP = {
        "Safe":          "rating_safe, ",
        "Questionable":  "rating_questionable, ",
        "Explicit":      "rating_explicit, ",
    }

    _SOURCE_MAP = {
        "Anime":   "source_anime, ",
        "Furry":   "source_furry, ",
        "Cartoon": "source_cartoon, ",
        "Pony":    "source_pony, ",
    }

    def generate(self, score="-", rating="-", source="-"):
        """Comnining whole string"""
        parts = []

        if score != "-":
            parts.append(self._SCORE_MAP.get(score, ""))

        if rating != "-":
            parts.append(self._RATING_MAP.get(rating, ""))

        if source != "-":
            parts.append(self._SOURCE_MAP.get(source, ""))

        result = "".join(parts)

        return (result,)
        
class ImageResizeNode:
    #ImageResizeNode is based on 🔧 Image Resize from https://github.com/cubiq/ComfyUI_essentials
    """
    Resize (compress/scale) an image with various methods.
    """

    # --------------------------------------------------------------------
    # UI interface (what the user sees in the graph)
    # --------------------------------------------------------------------
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION     = "execute"
    CATEGORY     = "essentials/image manipulation"

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the inputs that appear in the UI.
        """
        return {
            "required": {
                # types that will be visible to the user in the UI
                "image":          ("IMAGE",),

                # integers – 0 means “do not change” (you can set a specific size)
                "width":          ("INT", {"default": 512, "min": 0}),
                "height":         ("INT", {"default": 512, "min": 0}),

                # dropdown lists
                "method":         (["stretch",
                                      "keep proportion",
                                      "fill / crop",
                                      "pad"],),
                "interpolation":  (["nearest",
                                    "bilinear",
                                    "bicubic",
                                    "area",
                                    "nearest-exact",
                                    "lanczos"],),

                # when to apply resizing
                "condition":      (["always",
                                    "downscale if bigger",
                                    "upscale if smaller",
                                    "if bigger area",
                                    "if smaller area"],),
            }
        }

    # --------------------------------------------------------------------
    # Core logic
    # --------------------------------------------------------------------
    def execute(self,
                image: torch.Tensor,
                width: int,
                height: int,
                method: str = "stretch",
                interpolation: str = "nearest",
                condition: str = "always"):
        """
        Input:
          image       – (B,H,W,C)
          width,height – integers (0 → “don’t change”; can specify a size)
          method      – stretch / keep proportion / fill / crop / pad
          interpolation – interpolation method
          condition   – when to perform resizing

        Output: (IMAGE, new_width, new_height)
        """
        # -----------------------------
        # 1. Original image dimensions
        # -----------------------------
        _, oh, ow, _ = image.shape

        # -----------------------------
        # 2. Determine target size and possible padding / coordinates
        # -----------------------------
        if method == "keep proportion":
            ratio   = min(width / ow if width else float("inf"),
                          height / oh if height else float("inf"))
            new_w, new_h = round(ow * ratio), round(oh * ratio)
            width, height = new_w, new_h
            pad_left = pad_right = pad_top = pad_bottom = 0

        elif method == "pad":
            ratio   = min(width / ow if width else float("inf"),
                          height / oh if height else float("inf"))
            new_w, new_h = round(ow * ratio), round(oh * ratio)
            pad_left  = (width - new_w) // 2
            pad_right = width - new_w - pad_left
            pad_top   = (height - new_h) // 2
            pad_bottom= height - new_h - pad_top
            width, height = new_w, new_h

        elif method == "fill / crop":
            target_w, target_h = width if width else ow, height if height else oh
            ratio   = max(target_w / ow, target_h / oh)
            new_w, new_h = round(ow * ratio), round(oh * ratio)

            x  = (new_w - target_w) // 2
            y  = (new_h - target_h) // 2
            x2 = x + target_w
            y2 = y + target_h

            if x2 > new_w:   x -= (x2 - new_w)
            if x < 0:        x = 0
            if y2 > new_h:   y -= (y2 - new_h)
            if y < 0:        y = 0

            width, height = new_w, new_h

        else:      # stretch or any unknown method
            width  = width  if width  else ow
            height = height if height else oh
            pad_left = pad_right = pad_top = pad_bottom = 0

        # -----------------------------
        # 3. Resize condition (condition)
        # -----------------------------
        should_resize = (
            condition == "always" or
            ("downscale if bigger" == condition and (oh > height or ow > width)) or
            ("upscale if smaller" == condition and (oh < height or ow < width)) or
            ("bigger area" in condition and (oh * ow > height * width)) or
            ("smaller area" in condition and (oh * ow < height * width))
        )

        if should_resize:
            # Convert to format (B,C,H,W)
            out = image.permute(0, 3, 1, 2)

            # Choose interpolation mode
            if interpolation == "lanczos" and comfy is not None:
                out = comfy.utils.lanczos(out, width, height)
            else:
                kwargs = {"size": (height, width)}
                if interpolation in ("linear", "bilinear", "bicubic", "trilinear"):
                    # only for modes that support align_corners
                    kwargs["align_corners"] = False
                out = F.interpolate(out, mode=interpolation, **kwargs)

            # If this is pad – add borders (value 0 → black background)
            if method == "pad" and (pad_left or pad_right or pad_top or pad_bottom):
                out = F.pad(out,
                            (pad_left, pad_right, pad_top, pad_bottom),
                            mode='constant',
                            value=0)

            # If this is fill / crop – cut the center of the image
            if method == "fill / crop":
                out = out[:, :, y:y2, x:x2]

            out = out.permute(0, 2, 3, 1)   # back to (B,H,W,C)

        else:
            # If condition didn’t trigger – return original image
            out = image

        return out, width, height
