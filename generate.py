import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from callbacks import Iteratorize, Stream
import gradio as gr
import fire
import sys
import os
import torch
import re

CACHE_DIR="/mnt/scratch/users/hm2066/models/huggingface/"

if torch.cuda.is_available():
    device = "cuda"
else:
    raise ValueError("Cuda not available!")


def main(
        base_model: str = "",
        # lora_weights: str = "lora_comp_llama",
        load_8bit: bool = True,
        # prompt_template: str = "compositional_QA",
        server_name: str = "0.0.0.0",
        share_gradio: bool = True,
):
    
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=CACHE_DIR
            )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     torch_dtype = torch.float16
        # )

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            prompt,
            temperature=0.7,
            top_p=0.75, 
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
            stream_output=True,
            **kwargs
    ):
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens
        }
        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    # yield prompter.get_response(decoded_output)
                    yield decoded_output
            return  # early return for stream_output
        
        # No stream
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield output

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=4,
                label="Prompt",
                placeholder="Please enter your prompt",
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=10,
                label="Output",
            )
        ],
        title="Comp-Llama",
        description="LlaMA fine-tuned on modified QASC dataset.",  # noqa: E501
    ).queue().launch(server_name=server_name, share=share_gradio)
        
if __name__ == "__main__":
    fire.Fire(main)
