import dspy
import os
from vec_inf.api import VecInfClient

if __name__ == "__main__":
   client = VecInfClient()
   response = client.launch_model("Meta-Llama-3.1-8B-Instruct")
   job_id = response.slurm_job_id
   status = client.get_status(job_id)

   if status.status == ModelStatus.READY:
       print(f"Model ready at {status.base_url}")
   else:
       client.shutdown_model(job_id)
       exit()
    
    # setup DSPy
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
