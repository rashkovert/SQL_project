import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = pd.read_csv('files/counts_per_MAjob.csv')

# Ensure correct column names
assert {'job_location', 'job_title_short', 'salary_avg'}.issubset(df.columns), "Missing required columns."

# Order job titles by total salary descending
job_order = (
    df.groupby('job_title_short')['salary_avg']
    .sum()
    .sort_values(ascending=False)
    .index
)

# Make job_title_short a categorical type for ordering
df['job_title_short'] = pd.Categorical(df['job_title_short'], categories=job_order, ordered=True)

# Order locations by core salary descending
# core salary applies to the most ubiquitous jobs (Data Analyst, 'Data Scientist, Data Engineer)
core_jobs = ['Data Analyst', 'Data Scientist', 'Data Engineer']

df['core_salary'] = (
    df[df['job_title_short'].isin(core_jobs)]
    .groupby('job_location')['salary_avg']
    .mean()
)

loc_order = df['core_salary'].index.tolist()

df['job_location'] = pd.Categorical(df['job_location'], categories=loc_order, ordered=True)

fig = px.bar(
    df,
    x='job_location',
    y='salary_avg',
    color='job_title_short',
    title='Stacked Barplot of Average Salary by Location and Job Title',
    labels={'salary_avg': 'Average Salary', 'job_location': 'Location', 'job_title_short': 'Job Title'},
    color_discrete_sequence=px.colors.qualitative.T10
)

# fig.add_trace(go.Scatter(
#     x=df['job_location'],
#     y=df['core_salary'],
#     name='Average salary <br>(Data Analyst, Data Scientist, Data Engineer)',
#     mode='lines+markers',
#     line=dict(color='black', width=2)
# ))

fig.update_layout(barmode='stack', legend_title_text='Job Title')
fig.show()
