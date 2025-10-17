
# TODO: This scope is used to define prompt template

###################### Prompt for single image ######################


text_analysis_prompt_template_1 = """
Consider the following two images:
Image 1:
"""

text_analysis_prompt_template_2 = """
Image 2:
"""

text_analysis_prompt_template_3 = """
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is [{}]. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
"""

text_preference_prompt_template = """
Based on the text below to the questions:
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
{}

Is the goal better achieved in Image 1 or Image 2?
Reply a single line of 0 if the goal is better achieved in Image 1, or 1 if it is better achieved in Image 2.
Reply -1 if the text is unsure or there is no difference.
"""


###################### Prompt for sequence of images ######################

text_analysis_prompt_sequence_template_1 = """
Consider the following two short video clips:
**Note**: Each video clip contains {} images, which are key frames extracted from a video and are presented in the same order as the original footage.
Video 1:
"""

text_analysis_prompt_sequence_template_2 = """
Video 2:
"""

text_analysis_prompt_sequence_template_3 = """
1. What is shown in Video 1?
2. What is shown in Video 2?
3. The goal is [{}]. Is there any difference between Video 1 and Video 2 in terms of achieving the goal?
"""

text_preference_prompt_sequence_template = """
Based on the text below to the questions:
1. What is shown in Video 1?
2. What is shown in Video 2?
3. The goal is {}. Is there any difference between Video 1 and Video 2 in terms of achieving the goal?
{}

Is the goal better achieved in Video 1 or Video 2?
Reply a single line of 0 if the goal is better achieved in Video 1, or 1 if it is better achieved in Video 2.
Reply -1 if the text is unsure or there is no difference.
"""






# TODO: This scope used to define task description and note

task_description_dict = {
    "drawer-open-v2": "open the drawer",
    "sweep-into-v2": "place the green cube so that it lies on the square hole",
    "plate-slide-v2": "place the black plate so that it lies inside the goal",
    "door-open-v2": "open the door of the safe"
}

