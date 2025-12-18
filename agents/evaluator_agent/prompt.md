You are helping optimize task offloading decisions for an edge-cloud system following the paper's framework.

## Environment Configuration:

{env_details}

## Workflow DAG (N = {N} tasks):

{chr(10).join(task_details)}

## Optimization Parameters:

{json.dumps(params, indent=2)}

## Planner's Strategic Analysis:

{plan[:800]}

## Your Task:

Generate 3-5 intelligent candidate placement policies p = {{l_1, l_2, ..., l_{N}}} using ONLY these location IDs: {location_ids}

Provide candidate policies as lists: [l_1, l_2, ..., l_{N}]

## Concise Output Requirement

Return only:

- A <summary> (≤ 25 words) giving the main insight
- A <policies> section listing 3–5 candidate policies

Format:

<summary>...</summary>
<policies>
[p1, p2, ..., pN]
[p1, p2, ..., pN]
</policies>

Do NOT output chain-of-thought. Think internally only.
