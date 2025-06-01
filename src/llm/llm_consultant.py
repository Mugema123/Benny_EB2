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
    Please answer the following:

    A. Which students are at highest risk of dropping out, and why?
    B. What early intervention strategies should be prioritized for these students (e.g., mentoring, tutoring, family outreach)?
    C. How should staff or counselors use this information to act in the next 30 days?

    Your response must be structured as: 
    - Summary of key risk insights
    - Top 3 recommended interventions
    - Stakeholder actions (for teachers, counselors, or school leaders)

2. Resource Allocation & Policy Advising:
    Please provide:

    A. A breakdown of how the most important features influence dropout or performance risk.
    B. Suggestions on how school administrators should reallocate resources (e.g., increase tutoring budget, adjust staff roles).
    C. One policy recommendation that would reduce student risk or improve efficiency.

    Your answer should include:
    - Resource reallocation table (brief)
    - Policy rationale
    - Plain-language recommendation for school leaders

3. Admissions Strategy Optimization:
   
    Your task:

    1. Recommend how the admissions process could be adjusted to attract more resilient students.
    2. Suggest onboarding steps that increase student engagement and long-term retention.
    3. Identify any red flags schools should look for during admissions.

    Output format:
    - Admission strategy summary (3 bullet points)
    - Top 3 onboarding improvements
    - Early warning signs to watch for during student intake

Ensure your suggestions are specific, practical, and understandable to educational stakeholders.
"""

    # Send prompt to the LLM
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']
