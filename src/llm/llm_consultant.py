# src/llm/llm_consultant.py

import ollama

def consult_llm_with_metrics(metrics: dict, feature_columns: list, model_name='llama2'):
    # Format the metrics into a readable string
    metric_summary = "\n".join([f"{key}: {value:.2f}" for key, value in metrics.items()])

    # Convert list of features to string
    feature_list = ", ".join(feature_columns)

    # Build the dynamic prompt
    prompt = f"""
You are an AI educational consultant assisting with a school dropout prediction system.
The machine learning model uses the following input features:
{feature_list}

Here are the current model performance metrics:

{metric_summary}

Based on this information, provide strategic and practical advice in the following areas:

1. Dropout Prediction & Early Intervention:
   - How can this model help identify at-risk students early?
   - What features are most valuable for early warning?

2. Resource Allocation & Policy Advising:
   - How should school administrators and policymakers interpret these metrics?
   - What policies or resources could be optimized based on feature importance?

3. Admissions Strategy Optimization:
   - What aspects of admissions or onboarding can be adjusted to improve success and retention?
   - Which student traits might predict resilience or dropout risk?

Ensure your suggestions are specific, practical, and understandable to educational stakeholders.
"""

    # Send prompt to the LLM
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']
