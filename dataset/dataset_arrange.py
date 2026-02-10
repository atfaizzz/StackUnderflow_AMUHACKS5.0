import pandas as pd
import ast

df = pd.read_csv('all_job_post.csv')

def clean_skills(skills):
    if pd.isna(skills):
        return ""
    if isinstance(skills, str):
        skills = ast.literal_eval(skills)  # "['a','b','c']" → ['a','b','c']
    return " ".join(skills)

df['skills'] = df['job_skill_set'].apply(clean_skills)
df['job_role'] = df['job_title']

df_new = df[['skills', 'job_role']]
df_new.to_csv('skills_job_role.csv', index=False)
