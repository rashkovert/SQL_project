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
# core salary applies to the most ubiquitous jobs (Data Analyst, 'Data Scientist, Senior Data Engineer)
core_jobs = ['Data Analyst', 'Data Scientist', 'Senior Data Engineer']

core_salary_by_location = (
    df[df['job_title_short'].isin(core_jobs)]
    .groupby('job_location')['salary_avg']
    .mean()
)

df['core_salary'] = df['job_location'].map(core_salary_by_location)

loc_order = core_salary_by_location.sort_values(ascending=False).index

df['job_location'] = pd.Categorical(df['job_location'], categories=loc_order, ordered=True)

# Sort DataFrame by job_location to ensure correct bar order
df_sorted = df.sort_values('job_location')

fig = px.bar(
    df_sorted,
    x='job_location',
    y='salary_avg',
    color='job_title_short',
    title='Average Salary by MA Location and Job Title',
    labels={'salary_avg': 'Average Salary', 'job_location': 'Location', 'job_title_short': 'Job Title'},
    color_discrete_sequence=px.colors.qualitative.T10
)
fig.update_layout(
    title_font_size=24,
    title_x=0.5
)

core_df = df_sorted[['job_location', 'core_salary']].drop_duplicates().sort_values('job_location')
fig.add_trace(go.Scatter(
    x=core_df['job_location'],
    y=core_df['core_salary'],
    name='Average salary <br>(Data Analyst, Data Scientist, Senior Data Engineer)',
    mode='lines+markers+text',
    text=core_df['core_salary'].round(-3).astype(int).astype(str),
    textposition='top center',
    line=dict(color='black', width=2)
))

fig.update_layout(barmode='stack', legend_title_text='Job Title')
fig.show()
