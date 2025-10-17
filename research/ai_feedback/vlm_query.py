from typing_extensions import TypedDict
from pydantic import BaseModel
import os
import enum
import cv2
import json
import numpy as np

import traceback
import tenacity
from langchain.prompts import PromptTemplate

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from research.ai_feedback.text_prompt_library import *
from research.utils.utils import print_green, print_yellow, create_image_from_sequence
from PIL import Image


DEBUG = False
SLEEP_AFTER_TRY = 2  # seconds
MAX_TOKENS = 1024
MAX_RETRY = 3

SAFETY_SETTING = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


"""
    Utility functions for Google models
"""
def connect_google(
        client, engine, messages, temperature, max_tokens, top_p, response_mime_type=None, response_schema=None,
        frequency_penalty=None, presence_penalty=None
):
    if DEBUG:
        print_green(f'[INFO] Connecting to Google engine ...')
    response = client.generate_content(
        messages,
        generation_config=genai.GenerationConfig(
            candidate_count=1,
            top_p=top_p,
            max_output_tokens=max_tokens,
            temperature=temperature,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
        ),
        stream=False
    )
    response.resolve()
    return response


def get_google_response(
        end_when_error, max_retry,
        client, engine, messages, temperature, max_tokens,
        top_p, frequency_penalty=None, presence_penalty=None, verbose=False, **kwargs,
):
    vlm_output = None
    response_mime_type = kwargs.get('response_mime_type', None)
    response_schema = kwargs.get('response_schema', None)
    key = kwargs.get('key', None)

    try:
        r = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(max_retry),
            wait=tenacity.wait_fixed(SLEEP_AFTER_TRY),
            reraise=True
        )
        response = r.__call__(
            connect_google,
            client=client,
            engine=None,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
        )
        vlm_output = response.text

    except Exception as e:
        print_yellow(f'[ERROR] Google model error, key: {key}')
        # print_yellow(traceback.format_exc())
        if end_when_error:
            raise e
    return vlm_output


"""
    Abstract class for VLM models
"""
class VLMBase:
    def __init__(
            self,
            engine,
            max_tokens=MAX_TOKENS,
            temperature: float = 0,
            top_p: float = 1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            frame_encode_format=None,
            max_frame=1,
            verbose=False
    ):
        self.model_conn_func = None  # to be init by child class
        self.model_client = None  # to be init by child class
        self.get_response_func = None  # to be init by child class

        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.frame_encode_format = frame_encode_format
        self.max_frame = max_frame
        self.verbose = verbose
        self.key = None

    def get_response(self, messages, end_when_error=False, max_retry=MAX_RETRY, **kwargs):
        assert self.model_client is not None, f"model_client is None, did you initialize the model?"
        return self.get_response_func(end_when_error=end_when_error, max_retry=max_retry,
                                      client=self.model_client, engine=self.engine, messages=messages,
                                      temperature=self.temperature, max_tokens=self.max_tokens,
                                      top_p=self.top_p,
                                      frequency_penalty=self.frequency_penalty,
                                      presence_penalty=self.presence_penalty,
                                      **kwargs)

    def get_response_with_text(self, text, end_when_error=False, max_retry=MAX_RETRY, **kwargs):
        return self.get_response(text, end_when_error=end_when_error, max_retry=max_retry, key=self.key, **kwargs)

    def get_response_with_video(self, frames, text=None, end_when_error=False, max_retry=MAX_RETRY, **kwargs):
        messages = self._construct_messages_with_video(frames, text, **kwargs)
        return self.get_response(messages, end_when_error=end_when_error, max_retry=max_retry, key=self.key, **kwargs)

    def get_response_with_image(self, image, text=None, end_when_error=False, max_retry=MAX_RETRY, **kwargs):
        messages = self._construct_messages_with_image(image, text, **kwargs)
        return self.get_response(messages, end_when_error=end_when_error, max_retry=max_retry, key=self.key, **kwargs)

    def get_response_with_miscellany(self, prompt, end_when_error=False, max_retry=MAX_RETRY, **kwargs):
        return self.get_response(prompt, end_when_error=end_when_error, max_retry=max_retry, key=self.key, **kwargs)

    def _construct_messages_with_video(self, frames, text=None, **kwargs):
        """
        GENERAL FUNC: This function should combine the text prompt (by calling get_text_prompt) and the video frames
        """
        raise NotImplementedError

    def _construct_messages_with_image(self, image, text=None, **kwargs):
        """
        GENERAL FUNC: This function should combine the text prompt (by calling get_text_prompt) and the image
        """
        raise NotImplementedError


    def update_engine(self, engine, key):
        raise NotImplementedError


"""
    Google Models
"""
class GoogleModel(VLMBase):
    def __init__(
            self,
            engine="gemini-1.5-pro",
            max_tokens=1024,
            temperature: float = 0,
            top_p: float = 1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            frame_encode_format=None,
            max_frame=1,
            verbose=False
    ):
        super().__init__(
            engine,
            max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            frame_encode_format=frame_encode_format,
            max_frame=max_frame,
            verbose=verbose
        )

        self.get_response_func = get_google_response
        self.key = None

    def update_engine(self, engine, key=None):
        if key is not None:
            self.key = key
        self.engine = engine

        genai.configure(api_key=self.key)           # Config new API key
        self.model_client = genai.GenerativeModel(model_name=self.engine, safety_settings=SAFETY_SETTING,
                                                  )

    def _construct_messages_with_image(self, image, text=None, **kwargs):
        """
        GENERAL FUNC: This function should combine the text prompt (by calling get_text_prompt) and the image
        """
        messages = [image, text]
        return messages

    def construct_text_prompt_template(self):
        pass


    def get_text_analysis_prompt(self, env_name):
        task_description = task_description_dict[env_name]
        task_note = task_note_dict[env_name]

        text_prompt = self.text_analysis_prompt_template.format(task_description=task_description, task_note=task_note)
        return text_prompt


"""
    Key Management
"""
class KeyHolder:
    def __init__(self, key_list, n_processes=1):
        self.key_list = key_list
        self.n_processes = n_processes

        self.key_pool_for_each_process = [[] for _ in range(self.n_processes)]
        rank = 0
        for i in range(len(self.key_list)):
            self.key_pool_for_each_process[rank].append(self.key_list[i])
            rank += 1
            if rank >= n_processes:
                rank = 0

        self.global_counter = [0] * self.n_processes

    def get_key(self, rank=0):
        key = self.key_pool_for_each_process[rank][self.global_counter[rank]]
        self.global_counter[rank] += 1
        if self.global_counter[rank] >= len(self.key_pool_for_each_process[rank]):
            self.global_counter[rank] = 0

        return key


def get_feedback_from_vlm(
        key_holder,
        image_1,
        image_2,
        model_names=(None, None),
        env_name=None,
        rank=0,
):
    if len(image_1.shape) == 4:
        S, C, H, W = image_1.shape
    else:
        raise NotImplementedError
    assert S == 1

    image_size = H
    segment_len = S
    model_names = tuple(model_names)
    assert len(model_names) == 2, f"Confused selected models: {model_names}"

    ai_feedback = GoogleModel()
    prompt_image_1 = image_1[0].transpose(1, 2, 0)
    prompt_image_2 = image_2[0].transpose(1, 2, 0)

    text_analysis_prompt = [
        text_analysis_prompt_template_1,
        Image.fromarray(prompt_image_1),
        text_analysis_prompt_template_2,
        Image.fromarray(prompt_image_2),
        text_analysis_prompt_template_3.format(task_description_dict[env_name])
    ]

    """================================ Summarize using VLM ================================"""
    # image_prompt = create_image_from_numpy(sub_segment[0])
    # image_prompt = create_image_from_sequence(sub_segment, n_steps, timesteps[start], image_size)

    ai_feedback.update_engine(engine=model_names[0], key=key_holder.get_key(rank=rank))
    if DEBUG and rank == 0:
        # cv2.imshow(f"Image 1", cv2.cvtColor(np.array(prompt_image_1), cv2.COLOR_RGB2BGR))
        # cv2.imshow(f"Image 2", cv2.cvtColor(np.array(prompt_image_2), cv2.COLOR_RGB2BGR))
        cv2.imshow(f"Image", cv2.cvtColor(np.hstack([np.array(prompt_image_1), np.array(prompt_image_2)]), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    analysis = ai_feedback.get_response_with_miscellany(text_analysis_prompt, end_when_error=True)
    if DEBUG and rank == 0:
        print(analysis)

    """================================ Generating preference using LLM ================================"""
    class Choice(enum.Enum):
        PREFERENCE_IMAGE_1 = "0"
        PREFERENCE_IMAGE_2 = "1"
        PREFERENCE_NONE = "-1"

    text_preference_prompt = text_preference_prompt_template.format(task_description_dict[env_name], analysis)

    ai_feedback.update_engine(engine=model_names[1], key=key_holder.get_key(rank=rank))
    preference = ai_feedback.get_response_with_text(
        text_preference_prompt, end_when_error=True, response_mime_type="application/json", response_schema=Choice)

    if DEBUG and rank == 0:
        print(f"Preference: {preference}")

    preference = preference.strip().replace("'", "").replace('"', '')

    if preference == "0":
        return 0
    elif preference == "1":
        return 1
    else:
        return -1


def get_feedback_from_vlm_for_sequence(
        key_holder,
        image_1,
        image_2,
        model_names=(None, None),
        env_name=None,
        rank=0,
):
    if len(image_1.shape) == 4:
        S, C, H, W = image_1.shape
    else:
        raise NotImplementedError
    assert S > 1

    model_names = tuple(model_names)
    assert len(model_names) == 2, f"Confused selected models: {model_names}"

    ai_feedback = GoogleModel()
    prompt_image_1 = create_image_from_sequence(image_1.transpose(0, 2, 3, 1), image_1.shape[0], obs_size=200)
    prompt_image_2 = create_image_from_sequence(image_2.transpose(0, 2, 3, 1), image_2.shape[0], obs_size=200)

    text_analysis_prompt = [
        text_analysis_prompt_sequence_template_1.format(S),
        Image.fromarray(prompt_image_1),
        text_analysis_prompt_sequence_template_2,
        Image.fromarray(prompt_image_2),
        text_analysis_prompt_sequence_template_3.format(task_description_dict[env_name])
    ]

    """================================ Summarize using VLM ================================"""

    ai_feedback.update_engine(engine=model_names[0], key=key_holder.get_key(rank=rank))
    if DEBUG and rank == 0:
        img_1_show = prompt_image_1
        img_2_show = prompt_image_2
        cv2.imshow(f"VIDEO 1", cv2.cvtColor(img_1_show, cv2.COLOR_RGB2BGR))
        cv2.imshow(f"VIDEO 2", cv2.cvtColor(img_2_show, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    analysis = ai_feedback.get_response_with_miscellany(text_analysis_prompt, end_when_error=True)
    if DEBUG and rank == 0:
        print(analysis)

    """================================ Generating preference using LLM ================================"""
    class Choice(enum.Enum):
        PREFERENCE_IMAGE_1 = "0"
        PREFERENCE_IMAGE_2 = "1"
        PREFERENCE_NONE = "-1"

    text_preference_prompt = text_preference_prompt_sequence_template.format(
        task_description_dict[env_name], analysis
    )

    ai_feedback.update_engine(engine=model_names[1], key=key_holder.get_key(rank=rank))
    preference = ai_feedback.get_response_with_text(
        text_preference_prompt, end_when_error=True, response_mime_type="application/json", response_schema=Choice)

    if DEBUG and rank == 0:
        print(f"Preference: {preference}")

    preference = preference.strip().replace("'", "").replace('"', '')

    if preference == "0":
        return 0
    elif preference == "1":
        return 1
    else:
        return -1