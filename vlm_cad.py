import os, shutil
import json
import re
import trimesh
import pyrender
import subprocess
import numpy as np
from PIL import Image as image
import google.generativeai as genai
import cadquery as cq
from cadquery import exporters
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg
import vertexai
from vertexai.generative_models import GenerativeModel,GenerationConfig, Part, Image
from dotenv import load_dotenv

load_dotenv()
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
FREECAD_CMD_PATH = os.getenv("FREECAD_CMD_PATH")
GROUND_TRUTH_IMAGE_PATH = os.getenv("GROUND_TRUTH_IMAGE_PATH")

# if not GEMINI_API_KEY:
#     print("Error: GEMINI_API_KEY not found in environment variables.")
#     exit()

if not FREECAD_CMD_PATH or not os.path.exists(FREECAD_CMD_PATH):
    print(" Error: FREECAD_CMD_PATH is invalid or not found.")
    exit()


vertexai.init(project="vlm-query1", location="us-central1")
vlm_model = GenerativeModel(model_name="gemini-2.5-pro")
model = GenerativeModel(model_name="gemini-2.5-pro")

generation_config = GenerationConfig(
    temperature=0.2)
image_path = r"C:\Users\Y8664226\Downloads\vlm_query\demo1.png"  
try:
    img = Image.load_from_file(image_path)
except FileNotFoundError:
    print(f" Error: Image file not found at '{image_path}'")
    exit()


def clean_response(text):
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# def render_and_export_image(step_filepath, output_filepath="rendered.png"):
def render_and_export_image(step_filepath, output_dir="renders", output_file="rendered.png"):
    cad = cq.importers.importStep(step_filepath)

    safe_dir = os.path.join(os.path.expanduser("~"), "Documents", output_dir)
    os.makedirs(safe_dir, exist_ok=True)

    views = {
        "front": (0, 0, 0),
        "top": (90, 0, 0),
        "side": (0, 90, 0),
        "isometric": (45, 45, 0)
    }

    temp_files = []

    # Render each view
    for view_name, rotation in views.items():
        rx, ry, rz = rotation

        cad_view = cad.rotate((0, 0, 0), (1, 0, 0), rx)
        cad_view = cad_view.rotate((0, 0, 0), (0, 1, 0), ry)
        cad_view = cad_view.rotate((0, 0, 0), (0, 0, 1), rz)

        svg_path = os.path.join(safe_dir, f"{view_name}.svg")
        png_path = os.path.join(safe_dir, f"{view_name}.png")

        exporters.export(cad_view, svg_path)

        drawing = svg2rlg(svg_path)

        # Export PNG with high DPI
        renderPM.drawToFile(drawing, png_path, fmt="PNG", dpi=300)

        print(f" Rendered {view_name} view at {png_path}")
        temp_files.append(png_path)

    # Combine all rendered images horizontally
    images = [image.open(f) for f in temp_files]

    # Get max height and total width
    max_height = max(img.height for img in images)
    total_width = sum(img.width for img in images)

    # Create a new blank image
    combined = image.new("RGBA", (total_width, max_height), (255, 255, 255, 255))

    # Paste each image into the combined image
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the final combined image
    output_path = os.path.join(safe_dir, output_file)
    combined.save(output_path)

    print(f"\n Combined image saved at {output_path}")
    return output_path


prompt = """
You are an expert CAD engineer with extensive experience interpreting engineering drawings and creating fully detailed, accurate 3D models suitable for manufacturing, assembly, and simulation. Your task is to carefully analyze the provided 2D engineering drawing image and extract **all information** necessary to generate a complete and validated 3D CAD model in FreeCAD that can be exported as a .step file.

The extracted information must be generalized, precise, and exhaustive. Avoid assumptions or biases toward typical shapes or conventions. Every measurement, feature, relationship, and note — regardless of prominence, clarity, or layer — must be considered and documented.
Detailed Instructions – Follow Exactly as Written:**

Complete Geometry Extraction:
    Identify and describe every shape, form, and structure present in the drawing without assuming geometry types.
    Include primary bodies, as well as secondary details such as embossed or cutout features.
    Capture shapes even if partially obscured or faint.

Comprehensive Measurement Documentation:
    Record all dimensions, including but not limited to lengths, widths, heights, thicknesses, diameters (Φ), radii (R), angles, depths, and positions.
    Specify units (e.g., mm, cm), tolerances, and required precision wherever indicated.
    Include relative measurements (e.g., offset distances, alignments).

Feature Identification:
    Extract all features such as holes, slots, grooves, threads, chamfers, fillets, cutouts, embossments, ribs, patterns, bosses, etc.
    For each feature, document its exact size, location, orientation, depth, and relationship to other geometry.

Interrelationships and Assembly Context:
    Identify how different parts relate spatially and functionally.
    Document alignments, offsets, symmetry, concentricity, reference points, mating surfaces, and assembly relationships.
    Include any implicit relationships necessary for accurate modeling.

Annotations and Notes:
    Record every annotation, dimension text, instruction, material specification, surface finish, tolerance, and manufacturing note.
    Include visible, hidden, and auxiliary lines, as well as centerlines and reference marks.

Layering and Representation Awareness:
    Extract both visible and hidden geometry lines, annotations, and construction geometry.
    Do not overlook subtle or faint details that are relevant to modeling intent.

Avoid Assumptions and Generalizations:
    Do not guess shapes, sizes, or relations unless explicitly stated.
    Treat even small, unclear, or partial details as potentially critical and document them.

8. Validation for 3D CAD Modeling:
    Ensure that the extracted information is sufficient to build a fully accurate 3D model.
    Review and account for every dimension, feature, and relationship so that the output can be directly converted into FreeCAD Python code.

Formatting Instructions:
    Present the information in a structured, clear format suitable for direct use in code generation.
    Use numbered lists, categories, or tables where appropriate.
    Group related features together and separate distinct sections (geometry, dimensions, relationships, annotations).
Be concise but thorough; every piece of information must be actionable for model creation.

Important Notes:
    Every detail in the drawing must be recorded.
    Avoid assumptions; document only what is present or can be logically inferred.
    This output will be used as the direct input for Python code generation that produces a valid FreeCAD STEP file.
    Accuracy, clarity, and completeness are paramount.

Be systematic, exhaustive, and highly attentive in your analysis. This is a mission-critical task where missing even a small detail can result in an inaccurate model.

"""

print("\n Calling Gemini to analyze the image...")
vlm_response = vlm_model.generate_content([prompt, img], generation_config=generation_config)
print("\n--- Gemini Response ---")
print(vlm_response.text)
vlm_instructions = vlm_response.text.strip()

# json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
# if json_match:
#     json_string = json_match.group(0)
#     try:
#         instructions = json.loads(json_string)
#         print("\n Successfully extracted JSON instructions:")
#         print(json.dumps(instructions, indent=2))
#     except json.JSONDecodeError:
#         print(" Warning: Invalid JSON in response.")
#         exit()
# else:
#     print(" Warning: No JSON object found.")
#     exit()

def extract_python_code(raw_text):
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return raw_text.strip()

def extract_json(raw_text):
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return raw_text.strip()

def extract_step_filename(output_text):
    match = re.search(r'(\S+\.step)', output_text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None
code_gen_prompt = f"""

You are an expert FreeCAD Python programmer. You are provided with instructions for creating a 3D model based on a 2D engineering drawing. Use these instructions to write Python code that produces a fully valid STEP file representing the complete geometry, including all specified measurements and features.

Main Objective:  
The generated code must create a geometrically valid and complete STEP file that contains all details, measurements, positions, orientations, and features exactly as described in the instructions. The STEP file should be usable for visualization, simulation, and manufacturing purposes.

Detailed Requirements:
Geometry Creation:
     Create every shape, form, and feature exactly as described—holes, slots, grooves, fillets, chamfers, etc.
    Do not assume any dimensions, shapes, or relations unless explicitly stated in the instructions.
    All geometry must be created using appropriate FreeCAD classes and methods (e.g., Part.makeBox, Part.makeCylinder, etc.).
    All shapes must be added to the FreeCAD document with `doc.addObject("Part::Feature", "Name")`.
Boolean Operations Robustness:
    When performing cut operations (.cut()), ensure that the cutting object is oversized (e.g., ~20% larger in all directions) to prevent geometry errors due to coplanar faces.
    Position oversized objects correctly so they fully intersect the target geometry during cutting.

Order of Operations:
    First, create the base solids.
    Then apply additive (fuse) and subtractive (cut) operations.
    Apply finishing features (fillets, chamfers) only after completing all cuts and unions.

Validation and Recompute:
    After creating all shapes, call `doc.recompute()`.
    For every shape, check `shape.isValid()` and only include valid shapes for export.
    If a shape is invalid, print a descriptive error message indicating which feature failed and why.

Export Instructions:
    Collect all valid shapes into a list named `valid_shapes`.
    Export using `Part.export(valid_shapes, "model.step")`.
    If no valid shapes are present, print "No valid shapes to export. The STEP file was not created."

Accuracy and Completeness:
    All dimensions and placements must exactly reflect the instructions.
    Avoid skipping small details, rounding errors, or missing relationships.
    Annotate errors in geometry creation but do not guess missing information.

Code Output Format:
    Output only the raw Python code without explanations, comments, or markdown formatting.
    Do not output anything other than the required Python code.

**Final Notes:**
- The STEP file format is preferred because it preserves full geometry, dimensions, and relationships required for engineering applications. STL files are suitable for 3D printing but do not retain parametric or measurement information.
- Therefore, you must always export as a STEP file using `Part.export(valid_shapes, "model.step")`.
- Ensure that the final file is fully valid, contains all specified features, and can be opened and used in standard CAD applications without errors.

Instructions to follow:
{vlm_instructions}

Python Code:
"""

print("\nCalling Gemini to generate FreeCAD code...")
response = model.generate_content([code_gen_prompt, img], generation_config=generation_config)
raw_code_output = response.text

print("\n--- Raw Gemini Response ---")
print(raw_code_output)

clean_code = extract_python_code(raw_code_output)
print("\nExtracted FreeCAD Code:")
# print(clean_code)


def refinement(clean_code: str, error_traceback: str, vlm_instructions: str):
    refinement_prompt = f"""
You are an expert Python debugger with FreeCAD specialization.
image of the drawing is attached for your reference
The user's code failed. Analyze the code and the error traceback, then provide a corrected version.
dont change core of the deisgn have image and instruction for your reference and draft , rectify the error accordingly

Recompute and Validate:
    Call doc.recompute() after all shapes are created.
    For every shape, check shape.isValid() before including it in the export.

Export:
    Collect all valid shapes into a list.
    Export only the valid shapes using Part.export(valid_shapes, "model.step").
    If no valid shapes are found, print an error message and do not export.

Completeness:
    Ensure that all dimensions, positions, orientations, and relationships are modeled accurately.
    Avoid assuming default sizes or geometries.
    Every annotation and specification from the instructions must be implemented.

Export the geometry only if it is valid using the correct function:
    For FreeCAD: Part.export([valid_shapes], "model.step").

Print appropriate messages:
    Confirm successful export.
    Report if shapes were invalid or empty and skipped.

Avoid assumptions:
    Do not attempt to export a shape without verifying its geometry.
    Do not rely on default or missing geometry.


Original instructions:
{vlm_instructions}

Problematic code:
```python
{clean_code}

Error traceback:
{error_traceback}
````

Correct the specific error and ensure it aligns with the goal. Provide only the corrected Python code.
"""
    print("\n Calling Gemini to refine the code...")
    response = model.generate_content([refinement_prompt, img])
    raw_refined_code = response.text
    print("\n--- Raw Refinement Response ---")
    # print(raw_refined_code)
    return extract_python_code(raw_refined_code)


def semantic_refinement(initial_code: str, vlm_instructions: str , step_filename:str):
    current_code = initial_code
    rendered_image_path = render_and_export_image(step_filename)

    # B. Generate Verification Questions (based on Figure 7)
    print(" Generating verification questions...")
    question_gen_prompt = f"""
    You are expertCAD engineer and Python programmerWHO analyse the rendered 3d designs from the Freecad Code

    Here the instruction and 2D cad  enginnering drawings groundtruth image is used to generate a 3D design with help of FreeCAD code
    You will be given a instruction , ground truth image, along with the FreeCAD code 
    Your job is to provide enough number of questions that can be used to verify if the rendered design and the FreeCAD code matches and align with the Groundtruth image and the Instructions.
    The questions should be framed such that answering "No" implies a change is needed.

    Important Rules:
    1. Do not make up questions if you cannot generate at least 2 based on the instruction and other provided info.
    2. Only reference entities mentioned within the description.
    3. Do not ask about relative orientations like "right" or "left".

    *Instructions**:
    {vlm_instructions}

    **freeCAD code**:
    {current_code}
    **Generated Questions **:
"""
    response = vlm_model.generate_content([question_gen_prompt, img])
    print(response.text)
    question = response.text.strip()
    # try:

    #     questions = json.loads(clean_response(response.text))
    #     print("Generated Questions:")
    #     for q in questions:
    #         print(f"- {q}")
    # except (json.JSONDecodeError, TypeError):
    #     print("Failed to generate valid questions. Skipping refinement.")

    print("Answering questions with visual analysis...")
    qa_prompt = f"""
Your job is to answer this set of questions by comparing the object in the "rendered_image" with the object in the "ground_truth_image".
The "rendered_image" is the current version of the object, and the "ground_truth_image" is what the object should look like.

Rules:
1.Answer each question with one of three options: "Yes", "No", or "Unclear".
2.Provide a brief "Reasoning" for every answer.
3.Use "Unclear" if you are unsure or do not have enough information from the images.


Questions to Answer:
{question}

Output Format:
{{
"answers":
}}
"""
    response = vlm_model.generate_content([qa_prompt, rendered_image_path, GROUND_TRUTH_IMAGE_PATH])
    print(response.text)
    feedback = extract_json(response.text)
    try:
        answers_data = json.loads(feedback)['answers']
        print("Visual Analysis Complete:")
        for item in answers_data:
            print(f"  - Q: {item['question']}")
            print(f"    A: {item['answer']} ({item['reasoning']})")
    except (json.JSONDecodeError, KeyError):
        print("Failed to get valid answers from visual analysis. Skipping refinement.")

    # D. Generate Ameliorative Feedback
    issues = [item for item in answers_data if item['answer'] in ["No", "Unclear"]]
    if not issues:
        print("No issues found. The model is satisfactory!")
        return current_code

    print("Synthesizing corrective feedback...")


    feedback_gen_prompt = f"""
Based on the following analysis of a 3D object, generate a concise paragraph of actionable feedback to help a programmer correct the mistakes.

Rules:
1.The corrections should aim to make all answers "Yes".
2.Do not change parts of the object that are already correct.
3.Do not give feedback on image quality, orientation, or scale.
4.The feedback should be a summary of practical corrections.

Identified Issues:
{feedback}
Corrective Feedback Summary:
"""
    response = vlm_model.generate_content(feedback_gen_prompt)
    fin_feedback = response.text.strip()
    print("Generated Feedback:")
    print(feedback)

     # E. Refine the Code
    print("Refining the code based on feedback...")
    refinement_prompt = f"""
You are an expert CAD programmer refining a script based on visual feedback.
Your task is to rewrite a CadQuery script to incorporate the necessary corrections.


Previous Code Version:
```python
{current_code}
```

Corrective Feedback Summary:
{fin_feedback}

Instructions:
1. Carefully rewrite the entire Python script to apply the corrections.
2. Ensure the final object is still saved to "model.step"
3. Output only the raw, corrected Python code.

Refined Python Code:
"""
    response = model.generate_content([refinement_prompt, img])
    refined_code = clean_response(response.text)
    return extract_python_code(refined_code)

def validate_step_file_size(file_path="model.step", min_size=3000):
    if not os.path.exists(file_path):
        print("STEP file does not exist.")
        return False
    
    size = os.path.getsize(file_path)
    if size > min_size:
        print(f"STEP file size {size} bytes is acceptable.")
        return True
    else:
        print(f"STEP file size {size} bytes is too small.")
        return False


backup_file = "main1_backup.py"


max_attempts = 10
for attempt in range(max_attempts):
    print(f"\nAttempt {attempt + 1} of {max_attempts}")
    if not isinstance(clean_code, str) or clean_code.strip() == "":
        print(" Error: Clean code is invalid.")
        break

    script_file = "generated_freecad_script.py"
    if script_file:
        shutil.rmtree("generated_freecad_script.py", ignore_errors=True)
    with open(script_file, "w",encoding='utf-8') as f:
        f.write(clean_code)

    print(f"Saved script to {script_file}")

    print(f"\nExecuting {script_file} with FreeCADCmd...")
    result = subprocess.run([FREECAD_CMD_PATH, script_file], capture_output=True, text=True)
    print("--- Standard Output ---")
    print(result.stdout)
    print("--- Standard Error ---")
    print(result.stderr)

    if "exported" in result.stdout.lower():
        print("STEP file exported successfully.")
        step_filename = extract_step_filename(result.stdout)
        if validate_step_file_size(step_filename):
            if os.path.exists(step_filename):
                shutil.copyfile(step_filename, backup_file)
                print("Backup created.")

            clean_code = semantic_refinement(clean_code,vlm_instructions, step_filename)
            script_file = "generated_freecad_script.py"
            with open(script_file, "w",encoding='utf-8') as f:
                f.write(clean_code)
            refined_result = subprocess.run([FREECAD_CMD_PATH, "generated_freecad_script.py"], capture_output=True, text=True)
            if "exported" in refined_result.stdout.lower():
                print("Final STEP file exported successfully after semantic refinement.")
                break
            else:
                error_traceback = refined_result.stderr + " " + refined_result.stdout
                new_code = refinement(clean_code, error_traceback, vlm_instructions)

                if not isinstance(new_code, str) or new_code.strip() == "":
                    print("Refinement did not return valid code. Aborting.")
                    break   
                clean_code = new_code
        else:
            if os.path.exists(backup_file):
                shutil.copyfile(backup_file, step_filename)
                print("File restored from backup.")
            print("STEP file validation failed.")
            error_traceback = "STEP file validation failed.please refine the code to extract shape values in step file"+ " " + result.stdout+" "+ result.stderr
            new_code = refinement(clean_code, error_traceback, vlm_instructions)
            if not isinstance(new_code, str) or new_code.strip() == "":
                print("Refinement did not return valid code. Aborting.")
                break

            clean_code = new_code
            print(" Refinement applied, retrying...\n")

    else:
        print("STEP export failed or error encountered.")
        error_traceback = result.stderr+ " " + result.stdout
        print(error_traceback)
        new_code = refinement(clean_code, error_traceback, vlm_instructions)
        
        if not isinstance(new_code, str) or new_code.strip() == "":
            print("Refinement did not return valid code. Aborting.")
            break

        clean_code = new_code
        print(" Refinement applied, retrying...\n")


else:
    print("\n All attempts exhausted. Please review the code and input data.")


# if __name__ == "__main__":
#     main()


























# def convert_step_to_stl(FREECAD_CMD_PATH, step_file, output_stl="rendered.stl"):
#     script_content = f"""
# import FreeCAD
# import Part

# doc = FreeCAD.newDocument()
# shape = Part.read(r"{step_file}")
# obj = doc.addObject("Part::Feature", "Shape")
# obj.Shape = shape
# doc.recompute()
# Part.export([obj], r"{output_stl}")
#         """
#     script_file = "convert_to_stl.py"
#     with open(script_file, "w") as f:
#         f.write(script_content)

#     result = subprocess.run([FREECAD_CMD_PATH, script_file], capture_output=True, text=True)
#     print("Output:", result.stdout)
#     print("Error:", result.stderr)
#     if os.path.exists(output_stl):
#         print(f"Converted {step_file} to {output_stl}")
#         return output_stl
#     else:
#         print("Conversion failed.")
#         return None


# def render_stl_to_image(stl_file: str , output_image="rendered.png"):

#     # Load STEP
#     mesh = trimesh.load_mesh(stl_file, force='mesh')  # ensure mesh
#     if mesh.is_empty:
#         print("Error: Mesh is empty")
#         return
    
#     # Center the mesh
#     mesh.apply_translation(-mesh.centroid)

#     # Create scene
#     scene = pyrender.Scene()
#     mesh_node = pyrender.Mesh.from_trimesh(mesh)
#     scene.add(mesh_node)

#     # Add camera
#     camera = pyrender.PerspectiveCamera(yfov=0.7)
#     cam_distance = mesh.extents.max() * 2
#     cam_pose = trimesh.transformations.translation_matrix([0, -cam_distance, mesh.extents[2]/2])
#     scene.add(camera, pose=cam_pose)

#     # Add lights
#     light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
#     scene.add(light, pose=cam_pose)

#     # Render offscreen
#     r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080)
#     color, depth = r.render(scene)
#     r.delete()

#     from PIL import Image
#     img = Image.fromarray(color)
#     img.save(output_image)
#     print(f"STEP rendered to {output_image}")



# # Extract Python code from Gemini's response
# def extract_python_code(raw_text):
#     match = re.search(r"```(?:python)?\s*(.*?)\s*```", raw_text, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return raw_text.strip()

# # Compare rendered image with ground-truth image
# def images_similar(img1_path, img2_path, threshold=10):
#     try:
#         img1 = Image.open(img1_path).convert('L').resize((256, 256))
#         img2 = Image.open(img2_path).convert('L').resize((256, 256))
#         diff = ImageChops.difference(img1, img2)
#         histogram = diff.histogram()
#         total_diff = sum(i * count for i, count in enumerate(histogram))
#         print(f"Image difference score: {total_diff}")
#         return total_diff < threshold
#     except Exception as e:
#         print(f" Image comparison failed: {e}")
#         return False
