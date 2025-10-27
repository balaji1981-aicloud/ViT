import os
import re
import json
import subprocess
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from PIL import Image
import base64
import math # Added for dynamic camera

load_dotenv()
GROUND_TRUTH_IMAGE_PATH = os.getenv("GROUND_TRUTH_IMAGE_PATH")

BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
OUTPUT_RENDER = "C:/Users/Y8664226/Downloads/vlm_query/render2.png"

from ollama import Client
client = Client(host='http://10.87.60.30:11434')
model_name = "qwen2.5vl:72b"
# coder_model="llama4:scout"
coder_client = Client(host='http://localhost:11434')
coder_model="qwen3-coder:480b-cloud"
image_path = r"C:\Users\Y8664226\Downloads\vlm_query\659_1-1.jpg"
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

render_fix = """
import bpy
from mathutils import Vector
import math

def setup_scene():
    '''Deletes all objects from the scene to ensure a clean slate.'''
    if bpy.context.scene.objects:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

def auto_frame_model(padding=1.4, cam_dist_factor=1.0, offset=(0, 0, 0)):
    '''Frames all visible objects with a camera, supporting padding and positional offsets.'''
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    renderable_objects = [obj for obj in bpy.context.scene.objects if obj.type in ('MESH', 'CURVE', 'SURFACE', 'META', 'FONT') and not obj.hide_render]

    if not renderable_objects:
        print("[WARN] No renderable objects found to frame. Exiting framing.")
        return

    has_geometry = False
    for obj in renderable_objects:
        if not obj.bound_box:
            continue
        for i in range(8):
            v_global = obj.matrix_world @ Vector(obj.bound_box[i])
            min_co = Vector(min(min_co[j], v_global[j]) for j in range(3))
            max_co = Vector(max(max_co[j], v_global[j]) for j in range(3))
            has_geometry = True

    if not has_geometry:
        print("[WARN] Objects found, but they have no geometry to frame. Exiting.")
        return

    center = (min_co + max_co) / 2
    dimensions = max_co - min_co
    
    camera = bpy.context.scene.camera
    if not camera:
        cam_data = bpy.data.cameras.new('AutoFrameCam')
        camera = bpy.data.objects.new('AutoFrameCam', cam_data)
        bpy.context.collection.objects.link(camera)
        bpy.context.scene.camera = camera
    
    max_dim = max(dimensions.x, dimensions.y, dimensions.z)
    if max_dim == 0:
        max_dim = 1.0 
        
    fov = camera.data.angle if camera.data.angle > 0 else math.radians(50)
    distance = (max_dim / 2.0) / math.tan(fov / 2.0) * cam_dist_factor * padding
    camera.location = center + Vector((distance, -distance, distance * 0.75))
    
    camera.data.clip_start = 0.1
    camera.data.clip_end = distance + max_dim * 2
    
    if 'CameraTarget' in bpy.data.objects:
        target = bpy.data.objects['CameraTarget']
    else:
        bpy.ops.object.empty_add(type='SPHERE')
        target = bpy.context.active_object
        target.name = "CameraTarget"
    
    # --- NEW: APPLY POSITIONAL OFFSET ---
    # This shifts the camera's focus point for better composition.
    # The offset is relative to the object's own dimensions.
    offset_vector = Vector((
        offset[0] * dimensions.x,
        offset[1] * dimensions.y,
        offset[2] * dimensions.z
    ))
    target.location = center + offset_vector

    for c in camera.constraints:
        if c.type == 'TRACK_TO':
            camera.constraints.remove(c)
            
    track = camera.constraints.new(type='TRACK_TO')
    track.target = target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    
    if not any(obj.type == 'LIGHT' for obj in bpy.context.scene.objects):
        bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0, 0, 0))
        light = bpy.context.active_object
        light.rotation_euler = (math.radians(45), 0, math.radians(-45))
        light.data.energy = 3.0

def render_scene(output_path):
    '''Configures render settings and saves the final image.'''
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.film_transparent = True
    
    if scene.render.engine == 'CYCLES':
        scene.cycles.samples = 64
        scene.cycles.device = 'GPU'
    
    print(f"[RENDER] Starting headless render to: {output_path}")
    bpy.ops.render.render(write_still=True)
    print("[RENDER] Render finished.")
    bpy.ops.wm.quit_blender()
"""
render_tail = f"""
# --- MAIN EXECUTION BLOCK ---
import traceback

try:
    print("[INFO] Blender script: Executing auto_frame_model()...")
    auto_frame_model(padding=1.4) 
    
    print("[INFO] Blender script: Executing render_scene()...")
    render_scene(output_path="{OUTPUT_RENDER}")
except Exception as e:
    print(f"[ERROR] An error occurred inside Blender during final setup or render:")
    traceback.print_exc()
"""

def extract_python_code(raw_text):
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return raw_text.strip()

def save_code_to_file(code_str, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code_str)
    print(f"[INFO] Script saved: {filename}")

def run_blender_background(script_path):
    if not os.path.exists(BLENDER_PATH):
        print(f"[FATAL] Blender executable not found at: {BLENDER_PATH}")
        sys.exit(1)
        
    cmd = [
        BLENDER_PATH,
        "--background",
        "--factory-startup", # Ensures clean environment for each run
        "--python", script_path
    ]
    print(f"[INFO] Running Blender with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result

def compare_with_vlm(ground_truth_img_path: str, rendered_img_path: str):
    print("[INFO] Performing VLM visual comparison...")
    if not os.path.exists(rendered_img_path):
        print("[WARN] Rendered image not found for VLM comparison.")
        return {"match": False, "confidence_score": 0.0, "reasoning": "Render file not created.", "feedback": "The script failed to produce a render."}


    vlm_comparison_prompt = f"""
    You are a meticulous Quality Assurance engineer specializing in CAD models.
    Compare the 2D engineering drawing (image 1) with the 3D model render (image 2).

    Your task is to determine if the 3D model accurately represents all features from the 2D drawing.
    Pay close attention to overall shape, proportions, number of gear teeth, the central bore, and any keyways or slots.

    Respond ONLY with a JSON object in the following format:
    {{
    "match": boolean,
    "confidence_score": float,
    "reasoning": "A brief explanation of your findings.",
    "feedback": "If match is false, provide specific, actionable feedback for the developer to correct the 3D model script."
    }}
    """
    try:
        comp_images= []
        for path in [ground_truth_img_path, rendered_img_path]:
            with open(path, "rb") as img_file:
                comp_images.append(base64.b64encode(img_file.read()).decode("utf-8"))

        response = client.generate(model=model_name, prompt=vlm_comparison_prompt, images=comp_images, options ={"temperature":0.2})
        json_text = response['response'].strip()
        comparison_result = json.loads(json_text.replace("```json", "").replace("```", ""))
        print(f"[INFO] VLM Comparison Result: {comparison_result}")
        return comparison_result
    except (json.JSONDecodeError, Exception) as e:
        print(f"[ERROR] Failed to get a valid JSON response from VLM: {e}")
        return {"match": False, "confidence_score": 0.0, "feedback": f"Failed to parse the VLM's comparison response. Error: {e}", "reasoning": "JSON parse error."}


def vlm_code_refienement(clean_code:str, error_traceback:str, instructions:str):
    refinement_prompt = f"""
You are an expert Blender Python (bpy) programmer and a meticulous CAD visualization engineer.
Your task is to perform visual and dimensional refinement of an existing Blender model generation script.

The script below was used to generate a 3D assembly from a 2D engineering drawing.
A Visual Comparison Feedback (from the VLM) highlights specific mismatches between the rendered 3D output and the original drawing.

Your job is to:
Analyze, correct, and regenerate the Blender Python code so that the new model visually and dimensionally aligns with the original ground truth as closely as possible.

---

### CONTEXT INPUTS

Original Design Intent (for reference):
{instructions}

Existing Blender Python Code (to refine):
{clean_code}

Visual Feedback and Corrections (PRIORITY):
{error_traceback}

---

ACTION PLAN

1. Interpret the Feedback:
   - Carefully analyze the feedback and identify all stated issues (e.g., size differences, missing or misplaced features, incorrect alignments, proportions, or geometry types).
   - Treat these as precise correction tasks — e.g., “the bore is too small” means increasing the corresponding dimension in code.

2. Apply Targeted Refinements:
   - Modify only the necessary parts of the existing Blender Python script to correct the detected visual or dimensional mismatches.
   - Do **not** change the modeling structure, logic, or unrelated components unless they directly affect the feedback issue.
   - Ensure that the corrected geometry maintains proper relationships, proportions, and spatial positioning in the full assembly.

3. Maintain Overall Integrity:
   - Preserve the design intent, naming conventions, assembly logic, and transformations of the original script.
   - Do not simplify, remove, or replace geometry unless explicitly required to correct the described issue.
   - Ensure all components remain properly aligned, scaled, and assembled according to the original instructions.

4. Output Requirements:
   - Provide the **fully corrected Blender Python script**, complete and runnable.
   - The output must include all prior modeling elements (components, transformations, materials, lighting, and camera setup if present).
   - Do **not** include explanations, reasoning, or additional text — only output the corrected code.

CRITICAL RULES
- Focus solely on **visual and dimensional corrections**, not runtime debugging (unless the feedback itself is an error traceback).
- Ensure every modification contributes directly to **reducing the visual difference** between the 3D render and the original drawing.
- Maintain **exact spatial coherence** — the refined model must have all components in their correct positions, proportions, and orientations.
- Avoid any hallucinated features or assumptions not mentioned in the instructions or feedback.
- The final script must be complete, consistent, and executable without modification.

Output strictly:
The entire corrected Blender Python code, enclosed in a single code block, and nothing else.
"""

    refine_response =  coder_client.generate(model=coder_model, prompt=refinement_prompt)
    raw_refined_code = refine_response['response'].strip()
    return extract_python_code(raw_refined_code)
    
def refinement(clean_code: str, error_traceback: str, instructions: str):
    refinement_prompt = f"""
You are an expert Python debugger and 3D CAD engineer specializing in Blender's bpy scripting.
The user attempted to generate a Blender Python script that constructs a 3D model, but it produced an error or incorrect geometry.
Your goal is to analyze and fix the problem **without altering the original design logic or intent.**

### Context:
- Original design instructions (from VLM):
{instructions}

- User’s generated (problematic) code:
{clean_code}

- Error traceback or visual/structural correction feedback:
{error_traceback}

Your Objectives:
1. Diagnose and Correct:
   - Analyze the cause of the issue based on the traceback or feedback.
   - Correct only the problematic logic — syntax errors, API misuse, geometry construction issues, naming errors, object reference bugs, or assembly transformation mistakes.
   - Maintain consistency with the original intended geometry and relationships.

2. Preserve Design Intent:
   - Do not simplify, omit, or modify components or geometry described in the original instructions.
   - Preserve all defined relationships, positions, and dimensions.
   - Fix only the specific code parts causing the failure or producing incorrect results.

3. Precision & Validity:
   - Do **not** hallucinate new components, features, or relationships.
   - Ensure all created objects have valid, nonzero dimensions and correct Blender data types.
   - Validate all bpy operations (object creation, mesh linking, transformations, parenting, modifiers, etc.).
   - Remove any unused or invalid bpy references safely.

4. Reconstruction Rules:
   - Generate a **fully corrected, runnable Blender Python script**.
   - Include all necessary imports, scene cleanup, and setup at the top:
     ```python
     import bpy
     import math
     from mathutils import Vector
     bpy.ops.object.select_all(action='SELECT')
     bpy.ops.object.delete(use_global=False)
     ```
   - Assemble components individually, then correctly position them in the scene.
   - Ensure the script runs in a fresh Blender environment without manual edits.

5. Output Requirement:
   - Provide **only the fully corrected and complete Blender Python code** — no explanations, comments, or natural language output.
   - The corrected code must be syntactically valid, geometrically consistent, and executable without further modification.

Your task:
Identify the cause of failure and produce the **corrected, final Blender Python script** that achieves the intended 3D assembly exactly as per the original design instructions.
"""

    print("\n[INFO] Calling Coder Model for refinement...")
    response = coder_client.generate(model=coder_model, prompt=refinement_prompt, options = {"temperature":0.2})
    raw_refined_code = response['response'].strip()
    return extract_python_code(raw_refined_code)

# 1. VLM to get instructions
vlm_prompt = f"""
You are an expert CAD and mechanical design engineer with deep expertise in interpreting 2D engineering drawings and converting them into precise 3D assembly models.

Your task:
Analyze the provided 2D engineering drawing image that represents a complete design or assembly.
Extract a comprehensive, dimensionally accurate, and spatially consistent set of instructions that can be directly used to generate a 3D Blender Python script representing the full assembly.

Core Directives

1. strict Dimensional and Spatial Fidelity
   - The 3D model must replicate the original drawing *exactly* in terms of geometry, scale, proportions, and the spatial arrangement of all parts.
   - Maintain the precise **positions, alignments, orientations, and dimensions** of every component as they appear in the drawing.
   - No reordering, scaling, or repositioning is allowed. The final model must match the original layout without deviation.

2. Component-Level Analysis
   - Identify and describe each component or part in the drawing.
   - For every component, extract its geometric structure, measurable dimensions, form, and functional purpose as represented in the drawing.
   - Maintain clear separation between components but ensure their details remain consistent with the overall assembly.

3. Assembly-Level Integration
   - Describe how all components relate, connect, and fit together within the complete design.
   - Include clear information about positional relationships, alignments, and how components interact spatially within the overall assembly.
   - Ensure that these relationships result in a fully coherent and dimensionally accurate reconstruction of the complete model.

4. Global Design Consistency
   - Represent the entire structure as a unified model where all components occupy their correct relative positions.
   - Preserve overall shape, symmetry, proportions, and design logic exactly as in the source drawing.
   - Avoid assumptions or reinterpretations beyond what is visually or dimensionally evident in the input image.

5. Blender Scripting Readiness
   - Provide clear, structured, and unambiguous design instructions that can be directly translated into a 3D Blender Python script.
   - Maintain a consistent logical sequence that reflects the real-world spatial hierarchy and build order of the model.
   - Use language suitable for precise geometric reconstruction, avoiding vague or interpretive phrasing.

Output Requirements
- Deliver a structured, logically ordered, and complete set of modeling instructions.
- The instructions must enable reconstruction of the **exact 3D geometry, proportions, and positions** of every part as shown in the original drawing.
- Ensure all dimensional and spatial relationships are preserved without approximation.
- The final result should represent the **entire assembly or model as one coherent, correctly aligned structure**.

Your focus is on **accuracy, consistency, and full spatial integrity** across all types of models and assemblies.
"""


vlm_response = client.generate(model=model_name, prompt=vlm_prompt, images=[image_b64],  options ={"temperature":0.2})
print("VLM response:", vlm_response['response'])
vlm_instructions = vlm_response['response'].strip()
print(vlm_instructions)
print("[INFO] VLM instructions received.")

# 2. Coder Model to generate initial code
coder_prompt = f"""
vlm_spec : {vlm_instructions}

You are an expert Blender Python (bpy) programmer and mechanical design automation engineer.

Your task:
Generate a **complete, valid, and precise Blender Python script** that reconstructs the full 3D assembly exactly as described in the provided structured specifications.

CORE OBJECTIVE
- The script must generate a **fully accurate 3D model** that reflects every component, feature, and relationship described in the input.
- All components must be **modeled individually** and then **assembled together** to form the complete structure exactly as in the original design.

MODELING PRINCIPLES

1. Dimensional and Spatial Fidelity
   - Every component must strictly follow the specified **geometry, dimensions, and proportions**.
   - Maintain the **exact spatial arrangement, alignment, and orientation** between all components.
   - The final assembly must match the original design layout precisely — no random placement, scaling, or reordering.
   - Components must not overlap, shift, or appear jumbled. Their positioning must follow the defined assembly hierarchy and constraints.

2. Component Creation
   - Create each component as a distinct mesh object.
   - Use appropriate geometric operations based on the input (no assumptions or hallucinations).
   - Preserve all defined features such as openings, protrusions, recesses, holes, or contours exactly as described.
   - Ensure each component’s local coordinate system is consistent with the global assembly reference.

3. Assembly Construction
   - Once all components are created, assemble them by applying precise transformations as defined in the input.
   - Apply only the specified positional and relational transformations — no arbitrary adjustments.
   - Maintain accurate orientation, contact, and alignment between connected or interacting components.

4. Integrity and Precision
   - Do not introduce any components, dimensions, or relationships that are not described.
   - Do not create invalid, zero-dimension, or placeholder geometries.
   - Preserve all real-world measurement fidelity.
   - Every numerical and relational detail must be sourced directly from the input instructions.

5. Scripting Requirements
   - The output must be a **self-contained, directly runnable Blender Python script**.
   - Include necessary setup, imports, and scene cleanup.
   - Use clear, structured logic for defining, transforming, and assembling components.
   - Maintain clean code conventions and minimal but clear inline comments.
   - Output must be only valid Python code — no extra explanations or metadata.

FINAL GOAL
Produce Blender Python code that:
- Accurately models all components with their correct geometry, scale, and proportions.
- Strictly preserves the original **assembly positioning and orientation**.
- Builds the **complete assembly** as one coherent model with correct relationships.
- Is generalized to handle **any type of mechanical or geometric model**, regardless of complexity or shape type.
"""

print("[INFO] Step 2: Generating Blender Python script...")
blender_code_response = coder_client.generate(model=coder_model, prompt=coder_prompt, images=[image_b64],  options ={"temperature":0.2})
code = blender_code_response['response'].strip()
# print("Coder response:", blender_code_response['response']) # Too verbose
clean_code = extract_python_code(code)
print(clean_code)
print("[INFO] Blender code generated successfully.")

# 3. Setup for Iterative Loop
script_file = "generated_blender_script.py"
max_attempts = 20
ground_truth_image = image_path


# --- REVISED MAIN ITERATIVE LOOP ---
for attempt in range(max_attempts):
    print(f"\n{'='*20} Attempt {attempt + 1}/{max_attempts} {'='*20}")
    
    # Reconstruct the full script in every loop iteration
    final_script_code = render_fix + "\n" + clean_code + "\n" + render_tail
    
    save_code_to_file(final_script_code, script_file)
    result = run_blender_background(script_file)
    
    print("\n--- Blender Standard Output ---")
    print(result.stdout)
    print("\n--- Blender Standard Error ---")
    print(result.stderr)

    if result.stderr:
        error_traceback = result.stdout + "\n" + result.stderr
        print(f"[WARN] Blender execution failed or render file missing. Refining...")
        
        # Refine ONLY the model generation code
        mode_code = refinement(clean_code, error_traceback, vlm_instructions)
        mod_code = extract_python_code(mode_code)
        if mod_code == clean_code:
            print("[FATAL] Refinement did not change the code. Possible deadlock. Aborting.")
            break
        else:
            clean_code = mod_code

    else:
        vlm_feedback = compare_with_vlm(image_path, OUTPUT_RENDER)
        if vlm_feedback.get("match"):
            print(f"\n[SUCCESS] VLM confirmed visual match on attempt {attempt + 1}. Reason: {vlm_feedback.get('reasoning')}")
            break
        else:
            print("[WARN] VLM detected a visual mismatch. Refining code...")
            error_feedback = f"Visual Mismatch Reason: {vlm_feedback.get('reasoning')}\nFeedback: {vlm_feedback.get('feedback', '')}"
            feedback  = vlm_feedback.get('reasoning') + "\n" + vlm_feedback.get('feedback', '')
            print(feedback)
    
            mode_code = vlm_code_refienement(clean_code, feedback, vlm_instructions)
            mod_code = extract_python_code(mode_code)
            if mod_code == clean_code:
                print("[FATAL] Refinement did not change the code. Possible deadlock. Aborting.")
                break
            else:
                clean_code = mod_code

    if not isinstance(clean_code, str) or len(clean_code) < 50:
        print("[FATAL] Refinement returned invalid or empty code. Aborting.")
        break
else:
    print(f"\n[FAIL] All {max_attempts} attempts failed. Please review the initial prompts and final generated code.")
