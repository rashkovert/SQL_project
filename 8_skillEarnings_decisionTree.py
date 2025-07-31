# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


SKILL_COUNT = 0
N_STRONGEST_TREES_TO_VIEW = 1
SAVE = True

def main():
    
    global df, tree_df, tree_df_full, skills
        
    df, skills = make_df(skill_count=SKILL_COUNT)

    X = df[skills]
    y = df[['salary_year_avg']]*1e-5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define model and parameter grid
    model = DecisionTreeRegressor(random_state=42)
    param_grid = {
        'max_depth': [7, 8, 9, 10, 11, 12, 15],
        'min_samples_leaf': [10, 25, 30, 35, 40, 45, 50, 100]
    }

    # GridSearch with 5-fold CV using R²
    grid = GridSearchCV(model, param_grid, cv=5, scoring='r2', return_train_score=True)
    grid.fit(X_train, y_train)

    best_est = grid.best_estimator_
    best_tree = best_est.tree_
    
    
        
    # Predict and evaluate
    y_pred = best_est.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Test R² Score: {r2:.3f}")

    # --- Visualize top N_STRONGEST_TREES_TO_VIEW models by mean cross-validated score ---
    # Sort by mean_test_score descending
    top_indices = np.argsort(grid.cv_results_['mean_test_score'])[::-1][:N_STRONGEST_TREES_TO_VIEW]

    for i, idx in enumerate(top_indices):
        # Get best estimator with matching parameters
        params = grid.cv_results_['params'][idx]
        tree = DecisionTreeRegressor(random_state=42, **params)
        tree.fit(X_train, y_train)
        
        plt.figure(figsize=(15, 8))
        plot_tree(tree, feature_names=X.columns, filled=True, max_depth=10, rounded=True, fontsize=4, precision=3)
        plt.title(f"Tree #{i+1}: max_depth={params['max_depth']}, "
                f"min_samples_leaf={params['min_samples_leaf']}")
        plt.tight_layout()
        
        if SAVE:
            plt.savefig(f"plots/8_tree_{i}.pdf", bbox_inches='tight')
        plt.show(block=False)
        
        

    tree_df = pd.DataFrame(node_walk(node=0, path={'path':{}}, tree=best_tree, features = X.columns))

    tree_df_full, tree_df = tree_df_prepper(tree_df)
    
    tree_df_plotter(tree_df_full, 
                tree_df_full.columns.difference(tree_df.columns))
    
    input("Press Enter to exit...")



def make_df(skill_count=10):
    
    skills = (pd.read_csv('files/jobSkills_long.csv').query(f"count > {skill_count}"))['skills']
    
    jobs = pd.merge(
        pd.read_csv('csv_files_init/skills_dim.csv', 
                    usecols=['skill_id', 'skills'])
                    .query('skills in @skills'),
        pd.read_csv('csv_files_init/skills_job_dim.csv', 
                    usecols=['job_id', 'skill_id']), 
        on='skill_id', how='inner')

    df = (
        pd.merge(
            pd.read_csv('csv_files_init/company_dim.csv', 
                        usecols=['name', 'company_id'])
            .rename(columns={'name': 'company'}),
            pd.read_csv('csv_files_init/job_postings_fact.csv',
                        usecols=['job_id','company_id','job_title_short', 'salary_year_avg'])
            .query(f"job_title_short == 'Data Analyst'")
            .drop_duplicates(),
            on='company_id', how='inner')
        .merge(jobs, on='job_id', how='inner')
        .drop(columns=['skill_id', 'company_id', 'job_title_short'])
        .dropna(subset=['salary_year_avg'])
        .drop_duplicates()
        .assign(exists = 1)
        .pivot(index=['job_id', 'company', 'salary_year_avg'], columns='skills', values='exists').reset_index().fillna(0)
    )
    return df, skills

def branch_effect_size(left_node, right_node, tree):

    # Means
    mu_l = tree.value[left_node][0][0]
    mu_r = tree.value[right_node][0][0]

    # Sample sizes
    n_l = tree.n_node_samples[left_node]
    n_r = tree.n_node_samples[right_node]

    # Impurities (MSE)
    mse_l = tree.impurity[left_node]
    mse_r = tree.impurity[right_node]

    # Convert MSE to unbiased sample variance
    s2_l = mse_l * n_l / (n_l - 1)
    s2_r = mse_r * n_r / (n_r - 1)

    # Pooled standard deviation
    s_pooled = np.sqrt(((n_l - 1)*s2_l + (n_r - 1)*s2_r) / (n_l + n_r - 2))

    # Cohen's d
    d = (mu_r - mu_l) / s_pooled
    return d, mu_r, mu_l

def node_walk(node, path, tree, features):

    paths = [path]

    left = tree.children_left[node]
    if left != -1:
        right = tree.children_right[node]
        d, mean_r, mean_l = branch_effect_size(left, right, tree)
        
        path_left = {'value':mean_l, 'diff':mean_l-mean_r, 'd':-d, 'size':tree.n_node_samples[left], 'path': {**path['path'], features[tree.feature[node]]:'excluding'}}
        path_right = {'value':mean_r, 'diff':mean_r-mean_l, 'd':d, 'size':tree.n_node_samples[right], 'path': {**path['path'], features[tree.feature[node]]:'including'}}

        paths += list(node_walk(left, path_left, tree, features))
        paths += list(node_walk(right, path_right, tree, features))

    return paths

def tree_df_prepper(tree_df):
    # remove empty result (node 0)
    tree_df = tree_df[~tree_df['value'].isna()].copy()
    tree_df.reset_index(drop=True, inplace=True)
    
    tree_df['value'] = tree_df['value']*100000
    tree_df['diff'] = tree_df['diff']*100000
    
    # return quantified decision
    tree_df['last'] = tree_df['path'].apply(
        lambda d: {k: d[k]} if isinstance(d, dict) and d and (k := next(reversed(d))) else None
    )
    
    tree_df['last skill'] = tree_df['path'].apply(
        lambda d: next(reversed(d)) if isinstance(d, dict) and d else None
    )

    # annotate leaves distinguished by path size
    tree_df['leaf'] = tree_df['path'].shift(-1).str.len() <= tree_df['path'].str.len()

    tree_df_full = pd.concat([tree_df, pd.DataFrame(list(tree_df['path']))], axis=1).copy()
    tree_df_full = tree_df_full[~tree_df_full['value'].isna()]
    
    return tree_df_full, tree_df

def tree_df_plotter(tree_df_full, skills):

    # Skill status mapping
    skill_status_map = {'excluding': 2, 'including': 1, None: 0, np.nan: 0}
    skill_matrix = tree_df_full[skills].applymap(lambda x: skill_status_map.get(x, 0))

    # Rename continuous columns
    tree_df_full = tree_df_full.copy()
    tree_df_full.rename(columns={
        'value': 'mean salary',
        'diff': 'expected gain',
        'd': 'effect size (d)'
    }, inplace=True)

    # Combine skill matrix and metrics (including size)
    combined_matrix = pd.concat([
        skill_matrix,
        tree_df_full[['mean salary', 'expected gain', 'effect size (d)', 'size']]
    ], axis=1)

    # Colormaps
    cmap_skills_list = ['white', 'green', 'red']
    cmap_salary = sns.light_palette("blue", as_cmap=True)
    cmap_diff = sns.diverging_palette(10, 240, as_cmap=True)
    cmap_d = sns.diverging_palette(275, 150, s=80, l=55, as_cmap=True)
    cmap_size = sns.light_palette("orange", as_cmap=True)

    colormaps = []
    norm_data = combined_matrix.copy()
    norm_funcs = {}

    # Normalization
    for col in norm_data.columns:
        if col in skills:
            colormaps.append(cmap_skills_list)
        else:
            if col == 'mean salary':
                cmap = cmap_salary
            elif col == 'expected gain':
                cmap = cmap_diff
            elif col == 'effect size (d)':
                cmap = cmap_d
            elif col == 'size':
                cmap = cmap_size
            colormaps.append(cmap)

            if col == 'size':
                log_vals = np.log10(norm_data[col].replace(0, np.nan)).replace(-np.inf, np.nan)
                norm = Normalize(vmin=log_vals.min(), vmax=log_vals.max())
                norm_data[col] = norm(log_vals)
            else:
                norm = Normalize(vmin=norm_data[col].min(), vmax=norm_data[col].max())
                norm_data[col] = norm(norm_data[col])

            norm_funcs[col] = norm

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(len(norm_data.columns) * 0.36, len(norm_data) * 0.13))

    for i, col in enumerate(norm_data.columns):
        for j in range(len(norm_data)):
            val = norm_data.iloc[j, i]

            # Color logic
            if col in skills:
                status = skill_status_map.get(tree_df_full.iloc[j][col], 0)
                color = cmap_skills_list[status]
            else:
                color = colormaps[i](val if not np.isnan(val) else 0)

            # Draw box
            rect = plt.Rectangle((i, j), 1, 1, facecolor=color, edgecolor='white')
            ax.add_patch(rect)

            # Red outline if it's the last skill
            if col in skills and col == tree_df_full.iloc[j]['last skill']:
                outline = plt.Rectangle((i, j), 1, 1, facecolor='none', edgecolor='black', linewidth=2)
                ax.add_patch(outline)

            # Text inside continuous boxes
            if col not in skills:
                raw_val = tree_df_full.iloc[j][col]
                label = (
                    f"{raw_val:.0f}" if col not in ['effect size (d)']
                    else f"{raw_val:.2f}"
                )
                ax.text(i + 0.5, j + 0.5, label,
                        ha='center', va='center', fontsize=4,
                        color='black', fontweight='bold')

    # Leaf sidebar
    for j in range(len(tree_df_full)):
        color = 'black' if tree_df_full.iloc[j]['leaf'] else 'white'
        ax.add_patch(plt.Rectangle((-0.6, j), 0.2, 1, facecolor=color, edgecolor='none'))

    # Axes styling
    ax.set_xlim(0, len(norm_data.columns))
    ax.set_ylim(len(norm_data), 0)
    ax.set_xticks(np.arange(len(norm_data.columns)) + 0.5)
    ax.set_xticklabels(norm_data.columns, rotation=30, ha='center', fontsize=5)

    ax.tick_params(axis='both', length=0)
    ax.set_aspect(0.5)
    ax.tick_params(axis='x', which='both', labeltop=True, labelbottom=True, top=True, bottom=True)
    ax.xaxis.set_ticks_position('both')
    ax.set_title("Expected Salary Posting\nGiven Skill Inclusion/Exclusion", fontsize=14, pad=0)

    # Hide y ticks
    ax.set_yticks([])
    ax.set_yticklabels([])

    # ---------- Legends ----------
    skill_legend = [
        Patch(facecolor='red', label='excluding'),
        Patch(facecolor='green', label='including'),
        Patch(facecolor='white', label='missing')
    ]

    leaf_legend = [
        Patch(facecolor='black', label='leaf skill')
    ]

    outline_legend = [
        Patch(facecolor='none', edgecolor='black', linewidth=2, label='last skill')
    ]

    legend_items = skill_legend + leaf_legend + outline_legend
    ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, borderaxespad=0.)

    # ---------- Colorbars ----------
    if not SAVE:
        for idx, (col, cmap) in enumerate({
            'mean salary': cmap_salary,
            'expected gain': cmap_diff,
            'effect size (d)': cmap_d,
            'size': cmap_size
        }.items()):
            axins = inset_axes(
                ax,
                width="3%",
                height="17%",
                loc='lower left',
                bbox_to_anchor=(1.12, 0.65 - idx * 0.2, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0
            )
            ColorbarBase(axins, cmap=cmap, norm=norm_funcs[col], orientation='vertical')
            axins.set_title(col, fontsize=7, pad=4)

    # Grid overlay
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xticks(np.arange(len(norm_data.columns)+1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(norm_data)+1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', bottom=False, left=False)

    fig.tight_layout()

    if SAVE:
        plt.savefig("plots/8_tree_heatmap.pdf", bbox_inches='tight')

    plt.show(block=False)



if __name__ == "__main__":
    main()


# %%
