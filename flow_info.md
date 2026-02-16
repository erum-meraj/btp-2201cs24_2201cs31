# Complete Flow of Evaluation: From Planner to Final Policy

## High-Level Overview

```
┌──────────────┐
│   PLANNER    │ ← Uses LLM for strategic analysis
└──────┬───────┘
       │ Produces: Strategic plan (text)
       ▼
┌──────────────┐
│  EVALUATOR   │ ← Uses LLM + Mathematical Evaluation
└──────┬───────┘
       │ Produces: Optimal policy (numbers)
       ▼
┌──────────────┐
│   OUTPUT     │ ← Uses LLM for explanation
└──────────────┘
```

## Key Point: LLM vs Mathematical Evaluation

**CRITICAL DISTINCTION:**

- **LLM is used for SUGGESTING candidates** (heuristic guidance)
- **Mathematics is used for EVALUATING candidates** (computing actual cost U(w,p))

### Why This Matters

```python
# LLM CANNOT do this:
policy = [0, 1, 2, 1]
cost = llm.calculate_cost(policy)  # ❌ WRONG! LLM can't compute exact costs

# LLM CAN do this:
prompt = "Given this workflow, suggest promising policies"
suggestions = llm.think(prompt)  # ✓ CORRECT! LLM suggests candidates

# MATHEMATICS must do this:
cost = evaluator.total_offloading_cost(workflow, policy, env)  # ✓ Uses equations!
```

## Detailed Flow: Step by Step

### STEP 1: PLANNER AGENT (Pure LLM)

**Input:**

- Environment configuration (DR, DE, VR, VE)
- Workflow DAG (tasks, dependencies)
- Parameters (CT, CE, delta_t, delta_e)

**Process:**

```python
def planner.run(state):
    # LLM analyzes the problem
    prompt = f"""
    Analyze this offloading scenario:
    - Environment: {env}
    - Workflow: {workflow}
    - Mode: {params}

    Provide strategic guidance.
    """

    plan = llm.think_with_cot(prompt)  # Pure LLM reasoning
    return plan  # Text output
```

**Output:**

```
Strategic Plan (Example):
"This is a compute-intensive workflow with low data dependencies.
Recommendations:
- Heavy tasks (1, 4) should go to cloud for faster execution
- Light tasks (2, 3) can stay local to save energy
- Co-locate dependent tasks to minimize data transfer"
```

**LLM Role:** ✅ Strategic analysis
**Math Role:** ❌ None yet

---

### STEP 2: EVALUATOR AGENT - Part A (LLM for Suggestions)

**Input:**

- Strategic plan from Planner
- Environment, Workflow, Parameters

**Process - Substep 2.1: Get LLM Suggestions**

```python
def evaluator._get_llm_candidates():
    prompt = f"""
    Based on this strategic plan:
    {plan}

    And this environment:
    - 4 tasks
    - Locations: 0 (IoT), 1 (Edge), 2 (Cloud)

    Suggest 3-5 specific candidate policies.
    Format: [l1, l2, l3, l4]
    """

    llm_response = llm.think_with_cot(prompt)

    # LLM might suggest:
    # "Based on the analysis:
    #  Policy 1: [2, 0, 0, 2] - Heavy tasks on cloud
    #  Policy 2: [0, 1, 1, 2] - Balanced approach
    #  Policy 3: [2, 2, 2, 2] - All on cloud"

    return parse_policies(llm_response)
    # Returns: [(2,0,0,2), (0,1,1,2), (2,2,2,2)]
```

**LLM Role:** ✅ Suggests 3-5 "smart" policies
**Math Role:** ❌ None yet

---

### STEP 2: EVALUATOR AGENT - Part B (Generate All Candidates)

**Process - Substep 2.2: Generate Additional Candidates**

```python
def candidate_generator.generate_candidates():
    candidates = []

    # Add LLM suggestions (from above)
    candidates.extend(llm_candidates)  # 3-5 policies

    # Add memory-based (from past executions)
    candidates.extend(memory_candidates)  # 0-5 policies

    # Add systematic heuristics
    systematic = [
        (0,0,0,0),  # All local
        (1,1,1,1),  # All edge
        (2,2,2,2),  # All cloud
        (0,1,2,0),  # Round-robin pattern
        # ... etc
    ]
    candidates.extend(systematic)  # ~6-10 policies

    # NOW THE KEY DECISION:
    total_combos = 3^4 = 81  # 3 locations, 4 tasks

    if total_combos <= 10000:
        # EXHAUSTIVE: Add ALL possible combinations
        all_policies = [
            (0,0,0,0), (0,0,0,1), (0,0,0,2),
            (0,0,1,0), (0,0,1,1), (0,0,1,2),
            # ... all 81 combinations
            (2,2,2,2)
        ]
        candidates.extend(all_policies)

    return unique(candidates)
```

**Why Exhaustive Search?**

Without exhaustive search, we might have:

- 5 LLM suggestions
- 3 memory policies
- 8 systematic patterns
- **Total: 16 candidates**

But there are 81 possible policies! We might miss the optimal one.

With exhaustive search:

- **Total: 81 candidates (COMPLETE)**
- **Guaranteed to find the absolute best policy**

**LLM Role:** ❌ Not involved in this step
**Math Role:** ❌ Not yet (just generating combinations)

---

### STEP 2: EVALUATOR AGENT - Part C (Mathematical Evaluation)

**This is where the REAL evaluation happens!**

**Process - Substep 2.3: Evaluate All Candidates Mathematically**

```python
def utility_tool.find_best_policy(candidates):
    best_policy = None
    best_cost = infinity

    # For EACH candidate policy
    for policy in candidates:  # All 81 policies

        # MATHEMATICAL EVALUATION (NO LLM!)
        # Uses Equations 1-8 from the paper

        # 1. Create placement dict
        placement = {1: policy[0], 2: policy[1], 3: policy[2], 4: policy[3]}

        # 2. Compute Energy Cost (Equation 3-5)
        ED = compute_data_energy(workflow, placement, env)
        EV = compute_task_energy(workflow, placement, env)
        E = CE * (ED + EV)

        # 3. Compute Time Cost (Equation 6-7)
        delay_dag = build_delay_dag(workflow, placement, env)
        delta_max = find_critical_path(delay_dag)
        T = CT * delta_max

        # 4. Compute Total Cost (Equation 8)
        U = delta_t * T + delta_e * E

        # 5. Track best
        if U < best_cost:
            best_cost = U
            best_policy = policy
            print(f"New best: {policy} with cost {U}")

    return best_policy, best_cost
```

**Example Evaluation Output:**

```
Evaluating 81 candidate policies...
  ✓ New best: (0,0,0,0) with U(w,p) = 156.234
  ✓ New best: (2,0,0,2) with U(w,p) = 89.567
  ✓ New best: (2,1,1,2) with U(w,p) = 67.123
  ...

Best policy found: (2,1,1,2) with cost 67.123
```

**LLM Role:** ❌ ZERO involvement (pure mathematics)
**Math Role:** ✅ Computes EXACT cost using paper equations

---

### STEP 3: OUTPUT AGENT (LLM for Explanation)

**Input:**

- Optimal policy: [2, 1, 1, 2]
- Cost: 67.123
- Strategic plan

**Process:**

```python
def output.format_output():
    prompt = f"""
    Explain this optimal policy:

    Policy: {optimal_policy}
    Cost: {best_cost}
    Strategic Plan: {plan}

    Provide a clear explanation for the user.
    """

    explanation = llm.think_with_cot(prompt)
    return explanation
```

**Output:**

```
Optimal Policy Explanation:

The system determined that policy [2,1,1,2] achieves the lowest cost
of 67.123 units by:

1. Task 1 → Cloud (location 2): Heavy computation benefits from cloud
2. Task 2 → Edge (location 1): Moderate task with local data access
3. Task 3 → Edge (location 1): Co-located with Task 2 (dependency)
4. Task 4 → Cloud (location 2): Final aggregation on fast cloud

This balances execution speed (cloud) with data transfer costs (edge).
```

**LLM Role:** ✅ Explains the results in natural language
**Math Role:** ❌ None (just presenting results)

---

## Summary: LLM vs Math Throughout

| Stage                    | LLM Role              | Math Role                 | Output             |
| ------------------------ | --------------------- | ------------------------- | ------------------ |
| **Planner**              | ✅ Analyze problem    | ❌                        | Strategic text     |
| **Evaluator - Suggest**  | ✅ Suggest candidates | ❌                        | 3-5 policies       |
| **Evaluator - Generate** | ❌                    | ❌ Enumerate combinations | 81 policies        |
| **Evaluator - Evaluate** | ❌                    | ✅ Compute U(w,p)         | Best policy + cost |
| **Output**               | ✅ Explain results    | ❌                        | User-friendly text |

## Why We Need Exhaustive Search

### Scenario 1: Small Problem (4 tasks, 3 locations = 81 combinations)

**With exhaustive search:**

```
Candidates: All 81 policies
Evaluation: Check all 81
Result: GUARANTEED optimal policy [2,1,1,2] with cost 67.123
```

**Without exhaustive search (only LLM + heuristics):**

```
Candidates:
  - 5 LLM suggestions
  - 3 memory policies
  - 8 systematic patterns
  Total: 16 policies

Best found: [2,0,0,2] with cost 72.456

PROBLEM: We missed the optimal [2,1,1,2] (cost 67.123)!
         We're 5.333 units above optimal (7.9% worse)
```

### Scenario 2: Large Problem (10 tasks, 5 locations = 9,765,625 combinations)

**With exhaustive search:**

```
IMPOSSIBLE! Would take hours/days and crash the system
```

**Without exhaustive search (only LLM + heuristics):**

```
Candidates: 25-30 smart policies
Result: Near-optimal solution found quickly
```

## The Key Insight

```
┌─────────────────────────────────────────────┐
│         LLM is a SUGGESTION ENGINE          │
│         Math is an EVALUATION ENGINE        │
│                                             │
│  LLM suggests WHERE to search               │
│  Math evaluates HOW GOOD each solution is  │
└─────────────────────────────────────────────┘
```

### Example Flow

```python
# Step 1: LLM suggests candidates
llm_says = "Try these promising policies: [2,0,0,2], [0,1,1,2]"

# Step 2: Generate more candidates
all_candidates = llm_suggestions + heuristics + exhaustive
# = [2,0,0,2], [0,1,1,2], ..., [all 81 combinations]

# Step 3: Math evaluates EACH candidate
for policy in all_candidates:
    cost = compute_exact_cost(policy)  # Uses equations 1-8
    if cost < best_cost:
        best_policy = policy
        best_cost = cost

# Step 4: Return the mathematically proven best
return best_policy  # [2,1,1,2] with exact cost 67.123
```

## Why Can't LLM Evaluate Policies?

**LLM is bad at:**

```python
# LLM cannot accurately compute:
ED = sum(DE(l_i) * (sum(d_{j,i}) + sum(d_{i,k})))  # Equation 4
delta_max = longest_path(delay_DAG)                # NP-hard problem
U(w,p) = delta_t * T + delta_e * E                 # Complex formula

# LLM would give approximate/wrong answers
llm: "I think the cost is around 65-70?"  # ❌ UNRELIABLE
math: "The cost is exactly 67.123"        # ✓ EXACT
```

**LLM is good at:**

```python
# LLM can understand patterns:
"This task is compute-intensive, so cloud is better"
"These tasks have dependencies, so co-locate them"
"This is latency-sensitive, so use edge"

# These become CANDIDATE suggestions
# Then MATH evaluates them precisely
```

## Configuration Options

### Option 1: Keep Current (Recommended)

```python
# Small problems: Exhaustive (guaranteed optimal)
# Large problems: Heuristics (fast, near-optimal)
max_exhaustive = 10000
```

### Option 2: More Aggressive Exhaustive

```python
# Willing to wait longer for guaranteed optimal
max_exhaustive = 50000
```

### Option 3: Always Use Heuristics

```python
# Never do exhaustive search
max_exhaustive = 0  # Disable exhaustive

# Pros: Always fast
# Cons: May miss optimal solution even on small problems
```

### Option 4: Always Exhaustive (Not Recommended)

```python
# Try all combinations no matter what
max_exhaustive = float('inf')

# Pros: Always optimal
# Cons: WILL CRASH on large problems
```

## Bottom Line

**Exhaustive search is needed because:**

1. **LLM suggests candidates** (smart, but incomplete)
2. **Math evaluates candidates** (exact, but needs all candidates)
3. **For small problems**: We CAN check all combinations → Do it!
4. **For large problems**: We CAN'T check all → Use smart suggestions

The current implementation gives you the **best of both worlds**:

- Small problems: Guaranteed optimal solution
- Large problems: Fast, near-optimal solution

**The exhaustive search is what makes the solution PROVABLY OPTIMAL for small problems!**
