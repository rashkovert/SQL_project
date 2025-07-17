import pandas as pd
import numpy as np
import plotly.graph_objects as go

import openai
import getpass
import sys

from utils import fit_asymptotic_curve, space_labels

def main():
    DF = plot_and_manual_analysis()
    if USE_API:
        ChatGPT_label_manual_clusters()
    else:
        DF = load_ChatGPT_manual_cluster_labels(DF)

def plot_and_manual_analysis():
    # Parameters
    csv_file = 'files/jobSkills_long.csv'
    skill_column = 'skills'
    role_column = 'job_title_short'
    x_column = 'count' 
    y_column = 'avg_salary'            
    p_xaxis_title = "Skill count"
    p_yaxis_title = "Average salary (USD)"
    p_title = "Skill-associated salary vs skill commonality"
    #curve_param_y = 3
    #curve_param_x = 10
    #curve_param_y = int(curve_param_y)
    #curve_param_x = int(curve_param_x)

    # Load data
    df = pd.read_csv(csv_file)

    # Validate columns
    required = {x_column, y_column, role_column}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    # Loop through each role and create a separate plot
    for role in df[role_column].dropna().unique():
        df_role = df[df[role_column] == role].copy()

        # a, b = fit_asymptotic_curve(
        #     df_role.nlargest(curve_param_y, y_column).iloc[curve_param_y-1][x_column],
        #     df_role.nlargest(curve_param_y, y_column).iloc[curve_param_y-1][y_column],
        #     df_role.nlargest(curve_param_x, x_column).iloc[curve_param_x-1][x_column],
        #     df_role.nlargest(curve_param_x, x_column).iloc[curve_param_x-1][y_column])

        # df_role['curve'] = a + b / df_role[x_column]
        # df_role['label'] = np.where(df_role[y_column] > df_role['curve'],
        #                             df[skill_column], "")

        df_role['label'] = df_role[skill_column]
        
        df_role['text_positions'] = space_labels(
            df_role[x_column].values,
            df_role[y_column].values,
            df_role['label'].values
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_role[x_column],
            y=df_role[y_column],
            mode='markers+text',
            text=df_role['label'], 
            textposition=df_role['text_positions'],
            textfont=dict(size=9)
        ))
        
        # x_vals = sorted(df_role[x_column].dropna())
        # curve_vals = [a + b / x for x in x_vals if x != 0]

        # fig.add_trace(go.Scatter(
        #     x=x_vals,
        #     y=curve_vals,
        #     mode='lines',
        #     line=dict(color='red', dash='dash')
        # ))

        # Layout
        fig.update_layout(
            title=dict(
                text=f'<b>{p_title}: {role} postings, 2023</b>',
                font=dict(size=24),
                x=0.5
            ),
            xaxis_title=p_xaxis_title,
            yaxis_title=p_yaxis_title, 
            xaxis_type='log',
            showlegend=False
        )

        # ADDING MANUAL GROUPINGS TO PLOT
        if role == 'Data Analyst':
            lines = [
                {'x0': 20, 'x1': 20, 'y0': 90000, 'y1': 135000},
                {'x0': 500, 'x1': 500, 'y0': 90000, 'y1': 135000},
                {'x0': 20, 'x1': 5000, 'y0': 90000, 'y1': 90000},
                {'x0': 100, 'x1': 100, 'y0': 65000, 'y1': 90000},
                {'x0': 200, 'x1': 200, 'y0': 90000, 'y1': 135000}
                ]

            base_annotation = {
                'showarrow': False,
                'xref': 'x',
                'yref': 'y',
                'font': dict(family='Arial', size=14, color='black'),
                'bordercolor': 'black',
                'borderwidth': 1,
                'bgcolor': 'lightgray'
            }

            annotations = [
                {**base_annotation, 'x': 1200, 'y': 115000, 'text': "Required, valued<br>(essential languages/interfaces)"},
                {**base_annotation, 'x': 600,  'y': 75000,  'text': "Required, fundamental"},
                {**base_annotation, 'x': 55,  'y': 75000,  'text': "Niche or<br>weakly valued"},
                {**base_annotation, 'x': 300,  'y': 120000, 'text': "Common,<br>valued<br>(cloud)"},
                {**base_annotation, 'x': 70,  'y': 125000, 'text': "Uncommon, valued<br>(ML and ancillary<br>languages/packages)"}
            ]
            annotations = [
                {**ann, 'x': np.log10(ann['x'])} for ann in annotations
            ]

            shapes = [
            dict(type="line",
                x0=line['x0'], x1=line['x1'],
                y0=line['y0'], y1=line['y1'],
                line=dict(color="black", width=2)
            ) for line in lines]

            fig.update_layout(annotations=annotations)
            fig.update_layout(shapes=shapes)

        
        fig.show()

    # DEFINING MANUAL GROUPINGS
    df['Groupings_manual'] = np.select(
        [(df['count'] > 500) & (df['avg_salary'] > 90000),
        (df['avg_salary'] > 90000) & (df['count'] > 200) & (df['count'] < 500),  # condition 2
        (df['avg_salary'] > 90000) & (df['count'] > 20) & (df['count'] < 200),   # condition 3
        (df['count'] > 100) & (df['avg_salary'] < 90000),                        # condition 4
        ((df['count'] < 100) & (df['avg_salary'] < 90000)) | ((df['count'] < 20) & (df['avg_salary'] > 90000))  # condition 5
        ], range(1, 6), default=0)


    df['manual_names'] = df['Groupings_manual'].map(
        dict(
            enumerate(
                ["none",
                "Required, valued (essential languages/interfaces)",
                "Required, fundamental",
                "Niche or weakly valued",
                "Common, valued (cloud)",
                "Uncommon, valued (ML and ancillary languages/packages)"]
            )
        )
    )

    df.to_csv('files/job_skills_post5.csv', index=False)
    return df




def ChatGPT_label_manual_clusters():
    # Prompt for model
    print("Select a model to use:")
    models = ['gpt-4', 'gpt-4o', 'gpt-3.5-turbo']
    for i, m in enumerate(models, 1):
        print(f"{i}. {m}")
    choice = input("Enter number (default=1): ").strip() or "1"

    try:
        model = models[int(choice)-1]
    except (IndexError, ValueError):
        print("Invalid choice. Defaulting to gpt-4.")
        model = "gpt-4"

    # Prompt user for how to enter API key
    print("\nHow would you like to enter your OpenAI API key?")
    print("1. Manually (input hidden)")
    print("2. Paste / keyboard copy (input visible)")
    key_method = input("Choose 1 or 2 (default=1): ").strip() or "1"

    if key_method == "2":
        api_key = input("Paste your OpenAI API key: ").strip()
    else:
        api_key = getpass.getpass("Enter your OpenAI API key: ").strip()

    if not api_key:
        print("No API key provided. Exiting.")
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)

    # Load CSV
    csv_path = input("Enter path to CSV (default='files/job_skills_post5.csv'): ").strip() or "files/job_skills_post5.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Validate required columns
    if 'Groupings_manual' not in df.columns or 'skills' not in df.columns:
        print("CSV must contain 'Groupings_manual' and 'skills' columns.")
        sys.exit(1)

    # Group and prepare prompt
    grouped = df.groupby('Groupings_manual')['skills'].unique().to_dict()
    prompt = "Suggest a short, descriptive English label for each list of skills.\n\n"
    for group, skills in grouped.items():
        skill_list = ", ".join(skills[:10])
        prompt += f"Group {group}: {skill_list}\n"
    prompt += "\nRespond with one label per group, matching the format: Group 0: <label>"

    # Send to ChatGPT using new API interface
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that assigns short, clear category labels to skill groups."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        print("\n📋 Suggested Labels:\n")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"API call failed: {e}")
        sys.exit(1)

USE_API = False
# to avoid paywall, annotations were generated online
def load_ChatGPT_manual_cluster_labels(df, overwrite=True):
    df_anno = pd.DataFrame([
        {
            "Groupings_manual": 0,
            "ChatGPT_label": "Niche or Outdated Tools",
            "ChatGPT_explanation": "Likely less common, legacy, or narrowly used tools"
        },
        {
            "Groupings_manual": 1,
            "ChatGPT_label": "Core Analytics & BI Tools",
            "ChatGPT_explanation": "Essential data analysis languages and platforms"
        },
        {
            "Groupings_manual": 2,
            "ChatGPT_label": "Cloud & Data Warehouse Platforms",
            "ChatGPT_explanation": "Enterprise-scale data storage, cloud services, and query tools"
        },
        {
            "Groupings_manual": 3,
            "ChatGPT_label": "Big Data & ML Engineering Stack",
            "ChatGPT_explanation": "Tools for data pipelines, ML, distributed computing"
        },
        {
            "Groupings_manual": 4,
            "ChatGPT_label": "General Productivity & Legacy Stats Tools",
            "ChatGPT_explanation": "Office tools and older statistical software"
        },
        {
            "Groupings_manual": 5,
            "ChatGPT_label": "Specialized DevOps & Visualization Frameworks",
            "ChatGPT_explanation": "Special-purpose tools for search, graph DBs, scripting, frontend, etc."
        }
    ])
    df_anno = df_anno.merge(df, on="Groupings_manual", how="right")

    if overwrite:
        df_anno.to_csv('files/job_skills_post5.csv', index=False)

    return(df_anno)

if __name__ == "__main__":
    main()

