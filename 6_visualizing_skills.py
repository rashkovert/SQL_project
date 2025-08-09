
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

from utils import space_labels

SAVE = True

def main():
    DF_1 = plot_and_manual_analysis(csv_file = 'files/jobSkills_long.csv', plot=1)
    DF_2 = plot_and_manual_analysis(csv_file = 'files/jobSkills_full.csv', plot=2)

    input("Press Enter to exit...")


def plot_and_manual_analysis(csv_file, plot=1):
    # Parameters
    skill_column = 'skills'
    role_column = 'job_title_short'
    x_column = 'count'
    y_column = 'avg_salary'
    p_xaxis_title = "Skill count"
    p_yaxis_title = "Average salary (USD)"
    p_title = "Skill-associated salary vs skill commonality"

    # Load data
    df = pd.read_csv(csv_file)
    
    # filter an outlier manually
    df = df[df['skills'] != 'svn']

    required = {x_column, y_column, role_column}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    for role in df[role_column].dropna().unique():
        df_role = df[df[role_column] == role].copy()
        df_role['label'] = df_role[skill_column]
        #df_role['text_positions'] = space_labels(
        #    df_role[x_column].values,
        #    df_role[y_column].values,
        #    df_role['label'].values
        #)

        plt.figure(figsize=(12, 7))
        ax = sns.scatterplot(data=df_role, x=x_column, y=y_column, s=10)

        texts = []
        for _, row in df_role.iterrows():
            if row['label']:
                texts.append(
                    ax.text(row[x_column], row[y_column], row['label'], fontsize=5)
                )
                
        # adjust_text(
        # texts, ax=ax,
        # only_move={'points': 'y', 'text': 'y'},
        # arrowprops=dict(arrowstyle="-", color='gray', shrinkA=5),
        # expand_text=(1.01, 1.05),
        # expand_points=(1.01, 1.05),
        # force_text=0.1,
        # force_points=0.05,
        # lim=100)

        plt.xscale('log')
        plt.title(f"{p_title}: {role} postings, 2023", fontsize=16)
        plt.xlabel(p_xaxis_title)
        plt.ylabel(p_yaxis_title)

        if role == 'Data Analyst' and plot == 1:
            # Groupings
            df['Groupings_manual'] = np.select(
                [
                    (df['count'] > 500) & (df['avg_salary'] > 90000),
                    (df['avg_salary'] > 90000) & (df['count'] > 200) & (df['count'] < 500),
                    (df['avg_salary'] > 90000) & (df['count'] > 20) & (df['count'] < 200),
                    (df['count'] > 100) & (df['avg_salary'] < 90000),
                    ((df['count'] < 100) & (df['avg_salary'] < 90000)) | ((df['count'] < 20) & (df['avg_salary'] > 90000))
                ],
                range(1, 6), default=0
            )

            df['manual_names'] = df['Groupings_manual'].map(
                dict(enumerate([
                    "none",
                    "Required, valued (essential languages/interfaces)",
                    "Required, fundamental",
                    "Niche or weakly valued",
                    "Common, valued (cloud)",
                    "Uncommon, valued (ML and ancillary languages/packages)"
                ]))
            )

            df.to_csv('files/job_skills_post5.csv', index=False)

            # Draw lines
            lines = [
                {'x0': 20, 'x1': 20, 'y0': 90000, 'y1': 135000},
                {'x0': 500, 'x1': 500, 'y0': 90000, 'y1': 135000},
                {'x0': 20, 'x1': 5000, 'y0': 90000, 'y1': 90000},
                {'x0': 100, 'x1': 100, 'y0': 65000, 'y1': 90000},
                {'x0': 200, 'x1': 200, 'y0': 90000, 'y1': 135000}
            ]

            for line in lines:
                plt.plot([line['x0'], line['x1']], [line['y0'], line['y1']],
                         color='black', linewidth=1.5)

            # Text box annotations (on log x-scale)
            annotations = [
                {'x': 1200, 'y': 115000, 'text': "Permissive, valued\n(essential languages/interfaces)"},
                {'x': 600, 'y': 75000, 'text': "Permissive, fundamental"},
                {'x': 55, 'y': 75000, 'text': "Niche or\nweakly valued"},
                {'x': 300, 'y': 120000, 'text': "Common,\nvalued\n(cloud)"},
                {'x': 70, 'y': 125000, 'text': "Uncommon, valued\n(ML and ancillary\nlanguages/packages)"}
            ]

            for ann in annotations:
                ax.text(
                    ann['x'], ann['y'], ann['text'],
                    fontsize=10, ha='center', va='center',
                    bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.3')
                )

        #if role == 'Data Analyst' and plot == 2:


        plt.tight_layout()
        plt.show(block=False)

    return df


if __name__ == "__main__":
    main()
