import dspy
import os
from vec_inf.api import VecInfClient
import sys

if __name__ == "__main__":
    job_id = sys.argv[1]
    model_name = sys.argv[2]

    client = VecInfClient()
    status = client.get_status(job_id)

    if status.status == ModelStatus.READY:
        print(f"Model ready at {status.base_url}")
    else:
        client.shutdown_model(job_id)
        exit()
    
    # setup DSPy (add logic to change the model name)
    lm = dspy.LM("openai/meta-llama/Meta-Llama-3.1-8B-Instruct",
                 api_base=status.base_url,
                 api_key="",
                 model_type="chat")

    dspy.configure(lm=lm)

    # calling lm directly

    qa = dspy.ChainOfThought('question -> answer')

    # Run with the default LM configured with `dspy.configure` above.
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print(response.answer)

    client.shutdown_model(job_id)
