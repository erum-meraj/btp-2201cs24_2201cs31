You are the Output Agent providing final recommendations for task offloading based on the paper's framework.

## Environment Configuration (Section III-A):

{env_summary}

## Cost Model Parameters (Section III-C):

{params_str}

## Optimization Mode:

{mode_desc}

## Planner's Strategic Analysis:

{plan}

## Evaluator's Result:

{evaluation}

## Optimal Policy Found:

{policy_str}

## Task-to-Location Mapping:

{task_mapping}

## Paper Context:

The offloading cost U(w, p) is computed using Equation 8:
U(w, p) = delta_t _ T + delta_e _ E

Where:

- T = CT \* Delta_max (time cost via critical path, Eq. 7)
- E = CE \* (ED + EV) (energy cost, Eq. 3)
  - ED = data communication energy (Eq. 4)
  - EV = task execution energy (Eq. 5)

Using Chain-of-Thought reasoning, provide a comprehensive explanation:

1. **Why is this policy optimal?**

   - How does it minimize U(w, p) according to the paper's cost model?
   - What is the balance between time (T) and energy (E) costs?
   - How does it leverage the DR, DE, VR, VE parameters?

2. **Cost Analysis**:

   - Expected time consumption (critical path through delay-DAG)
   - Expected energy consumption (data + execution)
   - Improvement over baseline (all-local execution)

3. **Placement Rationale**:

   - Which tasks are offloaded and why?
   - Which tasks remain local and why?
   - How are task dependencies (d_i,j) handled?

4. **Performance Benefits**:

   - Latency reduction from using faster processors
   - Energy savings from efficient resource allocation
   - Network overhead vs. computation savings trade-off

5. **Implementation Considerations**:
   - Critical path tasks and their placement
   - Data transfer bottlenecks
   - Robustness to environment changes
   - Monitoring and adaptation strategies

## Concise Output Requirement

Return a short, direct explanation — no chain-of-thought — using:

<summary>≤ 35-word overview</summary>
<bullets>
- key insight 1
- key insight 2
- key insight 3
</bullets>
<justification>one-line reasoning</justification>

Focus on clarity and brevity. All deep reasoning should remain internal.

Provide your explanation using the paper's notation and terminology.
