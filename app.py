import gradio as gr
from datasets import load_dataset

import re
import os
import requests

from share_btn import community_icon_html, loading_icon_html, share_js

# TODO
#word_list_dataset = load_dataset("stabilityai/word-list", data_files="list.txt", use_auth_token=True)
#word_list = word_list_dataset["train"]['text']
word_list = []

def infer(prompt, negative="low_quality", scale=7):
    for filter in word_list:
        if re.search(rf"\b{filter}\b", prompt):
            raise gr.Error("Unsafe content found. Please try again with different prompts.")
        
    images = []
    url = os.getenv('JAX_BACKEND_URL')
    payload = {'prompt': prompt, 'negative_prompt': negative, 'guidance_scale': scale}
    images_request = requests.post(url, json = payload)
    for image in images_request.json()["images"]:
        image_b64 = (f"data:image/jpeg;base64,{image}")
        images.append(image_b64)
    
    return images, gr.update(visible=True)
    
    
css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .gradio-container {
            max-width: 730px !important;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; max-width: 13rem; margin-left: auto;}
        div#share-btn-container > div {flex-direction: row;background: black;align-items: center}
        #share-btn-container:hover {background-color: #060606}
        #share-btn {all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.5rem !important; padding-bottom: 0.5rem !important;right:0;}
        #share-btn * {all: unset}
        #share-btn-container div:nth-child(-n+2){width: auto !important;min-height: 0px !important;}
        #share-btn-container .wrap {display: none !important}
        #share-btn-container.hidden {display: none!important}
        
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #prompt-container .form{
        border-top-right-radius: 0;
        border-bottom-right-radius: 0;
        }
        #gen-button{
        border-top-left-radius:0;
        border-bottom-left-radius:0;
        }
        #prompt-text-input, #negative-prompt-text-input{padding: .45rem 0.625rem}
        #component-16{border-top-width: 1px!important;margin-top: 1em}
        .image_duplication{position: absolute; width: 100px; left: 50px}
"""

block = gr.Blocks(css=css)

examples = [
    [
        'A high tech solarpunk utopia in the Amazon rainforest',
        None,
        None
    ],
    [
        'A pikachu fine dining with a view to the Eiffel Tower',
        None,
        None
    ],
    [
        'A mecha robot in a favela in expressionist style',
        None,
        None
    ],
    [
        'an insect robot preparing a delicious meal',
        None,
        None
    ],
    [
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
        None,
        None
    ],
]


with block:
    gr.HTML(
        """
            <div style="text-align: center; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;margin-top:5px">
                  Stable Diffusion XL Demo
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">
                SDXL is a high quality text-to-image model from Stability AI. This demo on a <a style="text-decoration: underline;" href="https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e-and-a3-gpus-in-ga">Google Cloud TPU v5e</a>, to achieve efficient and cost-effective inference of 1024×1024 images. <a href="#">How does it work?</a>.
              </p>
            </div>
        """
    )
    
    with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt",
                        elem_id="prompt-text-input",
                )    
                btn = gr.Button("Generate", scale=0, elem_id="gen-button")

    gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery", grid=[2]
    )
    

    with gr.Group(elem_id="share-btn-container", visible=False) as community_group:
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("Share to community", elem_id="share-btn")
    
    with gr.Accordion("Advanced settings", open=False):
             negative = gr.Textbox(
                        label="Enter your negative prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter a negative prompt",
                        elem_id="negative-prompt-text-input",
             )
             guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=9, step=0.1
             )

    ex = gr.Examples(examples=examples, fn=infer, inputs=[text, negative, guidance_scale], outputs=[gallery, community_group], cache_examples=True)
    negative.submit(infer, inputs=[text, negative, guidance_scale], outputs=[gallery, community_group], postprocess=False)
    text.submit(infer, inputs=[text, negative, guidance_scale], outputs=[gallery, community_group], postprocess=False)
    btn.click(infer, inputs=[text, negative, guidance_scale], outputs=[gallery, community_group], postprocess=False)
        
    share_button.click(
            None,
            [],
            [],
            _js=share_js,
    )
    gr.HTML(
            """
                <div class="footer">
                    <p>Model by <a href="https://huggingface.co/stabilityai" style="text-decoration: underline;" target="_blank">StabilityAI</a> - backend running JAX on TPUs due to generous support of <a href="https://sites.research.google/trc/about/" style="text-decoration: underline;" target="_blank">Google TRC program</a> - Gradio Demo by 🤗 Hugging Face
                    </p>
                </div>
           """
    )
    with gr.Accordion(label="License", open=False):
            gr.HTML(
                """<div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a <a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md" style="text-decoration: underline;" target="_blank">Stability AI CreativeML Open RAIL++-M</a> license. The License allows users to take advantage of the model in a wide range of settings (including free use and redistribution) as long as they respect the specific use case restrictions outlined, which correspond to model applications the licensor deems ill-suited for the model or are likely to cause harm. For the full list of restrictions please <a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    <p><h4>Biases and content acknowledgment</h4>
Despite how impressive being able to turn text into image is, beware that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes. You can read more in the <a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0" style="text-decoration: underline;" target="_blank">model card</a></p>
               </div>
                """
            )
        

block.queue(concurrency_count=4, max_size=10).launch()
#block.launch(server_name="0.0.0.0")
