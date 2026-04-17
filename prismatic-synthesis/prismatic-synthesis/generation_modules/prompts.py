problem_instruction_template = """Given a set of example math problems, create $#$num_new_problems$#$ more similar or harder problems.

Each of the example should be formatted as:

---
[[Problem]]
your new problem - come up with a non-multiple choice problem, even if the provided examples are multiple choices. 
---

Examples:

"""

diverse_problem_instruction_template = """Given a set of example math problems, create $#$num_new_problems$#$ problems that are in similar area but of olympiad level.

First, analyze the characteristics of existing problems. 

Next, plan out how you will create $#$num_new_problems$#$ problems, including the area of problems (inspired by the example problems) and the problem variation strategies.

Then, create each problem. Each of the $#$num_new_problems$#$ problems should be formatted as:

---
[[Problem]]
your new problem - come up with a non-multiple choice problem, even if the provided examples are multiple choices. 
---

Examples:

"""

problem_fewshot_template = """---
[[Problem]]
$#$problem$#$
---"""

solution_instruction_system_template = """Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions.
This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, back-tracing, and iteration to develop well-considered thinking process.
"""

solution_instruction_template = """Solve the following problem. Make sure to put your final answer in \\boxed{}.

$#$problem$#$"""
