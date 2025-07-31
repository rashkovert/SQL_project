# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.stats import f_oneway, ttest_1samp
import pingouin as pg
import joblib

LOAD_CLUSTERS = True

def main():
    if not LOAD_CLUSTERS:
        print("Computing clusters for Data Analyst role...")
        df, umap_df, pca, pca_skill_order = compute_clusters(role='Data Analyst', skill_count=200, limit=0, N_PCs=12,  umap_n_neighbors=100, umap_min_dist=0.5, clst_min_samples=15, clst_eps=0.6, SAVE=True)
    else:
        umap_df = pd.read_csv('files/umap_results.csv')
        pca = joblib.load('files/pca_model.pkl')
        pca_skill_order = pd.read_csv('files/pca_skill_order.csv')
        df = pd.read_csv('files/skills.csv')
        
    pca_df = pd.read_csv('files/pca_results.csv')
    
    pca_df['salary_year_avg'] = umap_df['salary_year_avg']
    correlations = pca_df.corr()['salary_year_avg'].drop('salary_year_avg')
   
    not_skills={'UMAP1','UMAP2', 'job_id','company','entry_count','cluster','cluster_size', 'salary_year_avg'}
    
    skills = list(set(umap_df.columns).difference(not_skills))
    # sort skills by usage
    skills = list(umap_df[skills].mean().sort_values(ascending=False).index)
    
    #tune_clustering(umap_df)
    #plot_pca_heatmap(pca, pca_skill_order, correlations)
    loadings_df = pd.DataFrame(pca.components_.T, index=pca_skill_order['skills'])
    # ^ PC1 exclusively (anti-)correlates with income (slightly PC3)
    l_pc1 = loadings_df[0].reindex(loadings_df[0].abs().sort_values(ascending=False).index)
    l_pc3 = loadings_df[2].reindex(loadings_df[2].abs().sort_values(ascending=False).index)

    skills_salary = list(
        pd.concat([-l_pc1[l_pc1.abs() > 0.2], l_pc3[l_pc3.abs() > 0.2]])
        .sort_values(ascending=False)
        .index)
    
    print(skills)
    #plot_skills_hmap(df, skills)
    
    plot_skills_umap(umap_df, ['salary_year_avg']+skills_salary)
    
    plot_clusters(umap_df)
    
    plot_cluster_skills(umap_df, skills)
    plot_salary_umap(umap_df, REMOVE_OUTLIERS=3)
    
    
    plot_cluster_skills_box(umap_df, skills=skills, n_clusters=len(np.unique(umap_df['cluster'])), sort_by='salary_year_avg')
    
    plot_skills_anova(umap_df, skills)
    
    input("Press Enter to exit...")
    

def plot_skills_hmap(df, skills, REMOVE_OUTLIERS=3):
    
    if REMOVE_OUTLIERS:
        df = df[
            df['salary_year_avg'].between(
                df['salary_year_avg'].mean() - int(REMOVE_OUTLIERS) * df['salary_year_avg'].std(),
                df['salary_year_avg'].mean() + int(REMOVE_OUTLIERS) * df['salary_year_avg'].std()
                )
            ]
    # Append salary column for later
    df_skills = df[skills]
    
    df_with_salary = df[skills + ['salary_year_avg']]

    # Cluster only on skill data
    clustermap = sns.clustermap(df_skills, method='ward', metric='euclidean', cmap='viridis',
                                figsize=(8, 8), col_cluster=True, row_cluster=True)

    # Get reordered indices
    reordered_idx = clustermap.dendrogram_row.reordered_ind

    # Reorder original df_with_salary
    df_reordered = df_with_salary.iloc[reordered_idx]

    # Plot heatmap with salary added as a side color bar
    # Use seaborn.clustermap again with salary as row_colors

    # Create a color-mapped Series
    import matplotlib.colors as mcolors
    norm = plt.Normalize(df_with_salary["salary_year_avg"].min(), df_with_salary["salary_year_avg"].max())
    salary_colors = sns.color_palette("viridis", as_cmap=True)(norm(df_reordered["salary_year_avg"]))

    g=sns.clustermap(df_reordered.drop(columns='salary_year_avg'),
                row_cluster=False, col_cluster=False,
                row_colors=salary_colors,
                cmap="viridis", figsize=(8, 8),
                dendrogram_ratio=(0.08, 0.0001),   # effectively hides dendrograms
                cbar_pos=(0.01, 0.8, 0.025, 0.18)   # optional: reposition colorbar to avoid overlap
                )
    g.ax_heatmap.set_ylabel('Job postings')
    g.ax_heatmap.set_yticks([])

    plt.show(block=False)

    
def plot_skills_anova(umap_df, skills):
    # 1. Prepare data
    # Filter valid clusters (e.g., exclude noise from DBSCAN if -1 is used)
    valid_clusters = umap_df[umap_df['cluster'] != -1][skills + ['cluster', 'salary_year_avg']]

    # 2. Run ANOVA
    groups = [group['salary_year_avg'].dropna() for _, group in valid_clusters.groupby('cluster')]
    f_stat, p_value = f_oneway(*groups)

    print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.3g}")

    # 3. Compute per-cluster means and test significance
    cluster_means = valid_clusters.groupby('cluster')['salary_year_avg'].mean()
    overall_mean = valid_clusters['salary_year_avg'].mean()

    # T-test each cluster's salaries against overall mean
    significance = []
    for cluster_id, group in valid_clusters.groupby('cluster'):
        t_stat, p_val = ttest_1samp(group['salary_year_avg'].dropna(), overall_mean)
        significance.append((cluster_id, p_val))


    # Collect cluster IDs and p-values
    cluster_ids, p_values = zip(*significance)
    
    # Apply Benjamini–Hochberg correction
    results = pg.multicomp(list(p_values), method='fdr_bh', alpha=0.05)

    # Extract corrected p-values and rejection decisions
    pvals_corrected = results[1]
    reject_null = results[0]

    # Mark significant clusters after correction
    sig_clusters = {cluster_ids[i] for i, r in enumerate(reject_null) if r}
    
    # 4. Create DataFrame for plotting
    plot_df = cluster_means.reset_index()
    plot_df['significant'] = plot_df['cluster'].apply(lambda c: c in sig_clusters)
    plot_df = plot_df.sort_values(by='salary_year_avg', ascending=False).reset_index(drop=True)


    # 5. Plot barplot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='cluster', y='salary_year_avg', hue='significant', palette={True: 'red', False: 'gray'}, order=plot_df['cluster'])

    plt.title(f'Average Salary per Cluster\nANOVA p-value: {p_value:.2g}')
    plt.xlabel('Cluster')
    plt.ylabel('Average Salary')
    plt.legend(title='Significant', loc='upper right')
    plt.tight_layout()
    plt.show(block=False)
    
    
def tune_clustering(umap_df, min_samples=[5,8, 10, 15, 20, 25]):
    embedding = joblib.load('files/umap_model.pkl')
    for i in min_samples:
        clustering = DBSCAN(eps=0.6, min_samples=i).fit(embedding)
        umap_df["cluster"] = clustering.labels_
        umap_df['cluster_size'] = umap_df.groupby('cluster')['cluster'].transform('count')
    
        plot_clusters(umap_df)

def plot_pca_heatmap(pca, pca_skill_order, correlations):
    
    loadings_df = pd.DataFrame(pca.components_.T, index=pca_skill_order['skills'])
    loadings_df.columns = ['PC' + str(int(col)+1) for col in loadings_df.columns]
    loadings_df = loadings_df.pow(3)

    
    g = sns.clustermap(loadings_df, cmap='seismic', center=0,
                   cbar=False, figsize=(3, 7), col_cluster=False)
    
    #plt.subplots_adjust(left=-0.05, top=1)

    # Move x-axis to top
    g.ax_heatmap.xaxis.set_ticks_position('top')
    g.ax_heatmap.xaxis.set_label_position('top')
    

    # Get the default tick positions (already centered)
    # and apply skill labels in the correct (clustered) row order
    reordered = g.dendrogram_row.reordered_ind
    g.ax_heatmap.set_yticks(np.arange(len(loadings_df))+0.5)
    g.ax_heatmap.set_yticklabels(list(loadings_df.index[reordered]), rotation=0, va='center')
    g.ax_heatmap.tick_params(axis='y', labelsize=9)
    
    g.ax_heatmap.tick_params(axis='x', labelsize=8, rotation=45)

    # Hide dendrograms and colorbar
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.cax.set_visible(False)
    
    
    # Barplot on top (smaller)
    ax_bar = g.figure.add_axes([0.15, 0.9, 0.46, 0.07])  # tweak as needed

    # Plot barplot on inset axes
    ax_bar.bar(correlations.index, correlations.values, color='skyblue')
    ax_bar.set_ylim(-1, 1)
    ax_bar.set_title('Correlation with Salary', fontsize=8)
    ax_bar.tick_params(axis='x', rotation=45, labelsize=8)

    plt.show(block=False)

def plot_cluster_skills(umap_df, skills, div=None):
    skill_len = len(skills)
    if div is None: div = np.floor(skill_len**0.5).astype(int)
    
    fig, axes = plt.subplots(nrows=div, ncols=np.ceil(skill_len/div).astype(int), figsize=(3*np.ceil(skill_len/div), 2*div), sharex=True)
    axes = axes.flatten() if skill_len > div else [axes]
    
    for c in range(len(skills)):
        sns.scatterplot(data=umap_df, x="UMAP1", y="UMAP2", hue=skills[c], ax=axes[c], legend=False, s=10, alpha=0.5)
        axes[c].set_title(skills[c])
        
        #axes[c].set_xlabel('Skills')
        #axes[c].set_ylabel('Fraction of Skills')
        #axes[c].tick_params(axis='x', rotation=90, labelbottom=True)
    
    for ax in axes[skill_len:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show(block=False)

    
def plot_cluster_skills_vln(umap_df, skills, n_clusters=16, div=None):
    if div is None: div = np.ceil(n_clusters**0.5).astype(int)
    # Show violin plots of skill frequencies for each cluster
    
    global top_clusters, c
    
    top_clusters = (
        umap_df[umap_df.cluster != -1][['cluster', 'cluster_size', 'salary_year_avg']]
        .groupby('cluster')
        .agg({'cluster_size': 'count', 'salary_year_avg': 'mean'})
        .reset_index()
        .sort_values('cluster_size', ascending=False)
        .head(n_clusters)
    )
    
    fig, axes = plt.subplots(nrows=np.ceil(n_clusters/div).astype(int), ncols=div, figsize=(4*np.ceil(n_clusters/div), 60/div), sharex=True)
    axes = axes.flatten() if n_clusters > div else [axes]
    for c in range(len(top_clusters['cluster'])):
        cluster = top_clusters['cluster'].iloc[c]
        cluster_size = top_clusters['cluster_size'].iloc[c]
        cluster_salary = top_clusters['salary_year_avg'].iloc[c]
        cluster_data = umap_df[umap_df['cluster'] == cluster][skills]
        cluster_data = cluster_data.melt(var_name='skills', value_name='fraction')
        
        sns.violinplot(data=cluster_data, x='skills', y='fraction', ax=axes[c], density_norm='width', inner='quartile')
        axes[c].set_title(f"Cluster {cluster}\nCompanies: {cluster_size}\nAvg Salary: {cluster_salary:3f}", fontsize=8)
        axes[c].tick_params(axis='x', rotation=90, labelbottom=True, labelsize=4)
        
    for ax in axes[len(top_clusters):]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show(block=False)

def plot_cluster_skills_box(umap_df, skills, n_clusters=16, div=None, sort_by='salary_year_avg'):
    if div is None:
        div = np.ceil(n_clusters**0.5).astype(int)

    top_clusters = (
        umap_df[umap_df.cluster != -1][['cluster', 'cluster_size', 'salary_year_avg']]
        .groupby('cluster')
        .agg({'cluster_size': 'count', 'salary_year_avg': 'mean'})
        .reset_index()
        .sort_values(sort_by, ascending=False)
        .head(n_clusters)
    )

    fig, axes = plt.subplots(
        nrows=np.ceil(n_clusters/div).astype(int),
        ncols=div,
        figsize=(4*np.ceil(n_clusters/div), 60/div),
        sharex=True
    )

    axes = axes.flatten() if n_clusters > div else [axes]

    for c in range(len(top_clusters['cluster'])):
        cluster = top_clusters['cluster'].iloc[c]
        cluster_size = top_clusters['cluster_size'].iloc[c]
        cluster_salary = top_clusters['salary_year_avg'].iloc[c]
        cluster_data = umap_df[umap_df['cluster'] == cluster][skills]
        cluster_data = cluster_data.melt(var_name='skills', value_name='fraction')

        sns.boxplot(data=cluster_data, x='skills', y='fraction', ax=axes[c], showfliers=False)
        axes[c].set_title(
            f"Cluster {cluster}\nRoles: {cluster_size}\nAvg Salary: {cluster_salary:.0f}",
            fontsize=8
        )
        axes[c].tick_params(axis='x', rotation=90, labelbottom=True, labelsize=4)
        if c % div == 0:
            axes[c].set_ylabel('Fraction')
        else:
            axes[c].set_ylabel('')
        axes[c].tick_params(axis='y', labelsize=6)

    for ax in axes[len(top_clusters):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show(block=False)


def plot_clusters(umap_df):
    fig2 = plt.figure(figsize=(5, 5))
    sns.scatterplot(data=umap_df, x="UMAP1", y="UMAP2", hue="cluster", palette="tab10", s=30, alpha=0.7)
    plt.title("UMAP with DBSCAN clustering")
    plt.xticks([]); plt.yticks([])
    
    for label in np.unique(umap_df["cluster"]):
        if label == -1:
            continue
        cluster_data = umap_df[umap_df['cluster'] == label]
        x_mean = cluster_data['UMAP1'].mean()
        y_mean = cluster_data['UMAP2'].mean()
        plt.text(x_mean, y_mean, str(label), fontsize=3, weight='bold', ha='center', va='center', bbox=dict(boxstyle='circle,pad=0.3', fc='white', ec='black'))
        
    plt.legend().remove()
        
    plt.show(block=False)
    

def plot_salary_umap(umap_df, REMOVE_OUTLIERS=3, AVG=False):
    
    if REMOVE_OUTLIERS:
        umap_df = umap_df[umap_df['salary_year_avg'].between(
            umap_df['salary_year_avg'].mean() - int(REMOVE_OUTLIERS) * umap_df['salary_year_avg'].std(),
            umap_df['salary_year_avg'].mean() + int(REMOVE_OUTLIERS) * umap_df['salary_year_avg'].std()
)]
    
    umap_df['salary_avg'] = umap_df.groupby('cluster')['salary_year_avg'].transform('mean')
    
    fig2 = plt.figure(figsize=(8, 8))
    sns.scatterplot(data=umap_df, x="UMAP1", y="UMAP2", hue='salary_avg' if AVG else 'salary_year_avg', s=30, alpha=0.7)
    plt.xticks([]); plt.yticks([])
    #plt.colorbar(label='Average Salary per Year')

    plt.show(block=False)

def plot_skills_umap(umap_df, skills, div=None):
    
    n_skills = len(skills)
    if div is None:
        div = np.ceil(n_skills**0.5).astype(int)

    plot_cols = div  # Adjust columns as neededimport matplotlib.pyplot as plt

    cols = plot_cols + 1  # Extra column for legend on the left
    rows = int(np.ceil(n_skills / plot_cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2), squeeze=False)

    skill_idx = 0
    first_handles, first_labels = None, None

    for row in range(rows):
        for col in range(1, cols):  # Start from column 1 to leave column 0 for legend
            if skill_idx >= n_skills:
                axes[row][col].axis('off')
                continue

            skill = skills[skill_idx]
            ax = axes[row][col]

            plot_df = umap_df[['cluster', 'UMAP1', 'UMAP2'] + [skill]].copy()
            plot_df[skill] = plot_df.groupby('cluster')[skill].transform('mean')

            scatter = sns.scatterplot(
                data=plot_df,
                x="UMAP1",
                y="UMAP2",
                hue=skill,
                s=10,
                alpha=0.7,
                ax=ax,
                palette="viridis"
            )

            if skill_idx == 0 and scatter.legend_:
                first_handles, first_labels = scatter.get_legend_handles_labels()
            
            ax.set_title(skill)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend_.remove()

            skill_idx += 1

        # Add legend to column 0 of the first row only
        if row == 0 and first_handles:
            legend_ax = axes[row][0]
            legend_ax.axis("off")
            legend_ax.legend(first_handles, first_labels, title="Skill Level", loc="center")

    # Hide all unused axes
    for r in range(rows):
        for c in range(cols):
            if (r * plot_cols + (c - 1)) >= n_skills and c > 0:
                axes[r][c].axis("off")
            if c==0 and r != 0:
                axes[r][c].axis("off")

    plt.tight_layout()
    plt.show(block=False)


def compute_clusters(role='Data Analyst', skill_count = 200, limit = 0, N_PCs = 10, umap_n_neighbors = 50, umap_min_dist = 0.5, clst_min_samples = 50, clst_eps=0.5, SAVE=True):
    
    skills = (pd.read_csv('files/jobSkills_long.csv').query(f"count > {skill_count}"))['skills']

    # Extract associated job IDs for these skills
    jobs = pd.merge(
        pd.read_csv('csv_files_init/skills_dim.csv', 
                    usecols=['skill_id', 'skills'])
                    .query('skills in @skills'),
        pd.read_csv('csv_files_init/skills_job_dim.csv', 
                    usecols=['job_id', 'skill_id']), 
        on='skill_id', how='inner')

    global df, df2
    # Load company data
    df = (
        pd.merge(
            pd.read_csv('csv_files_init/company_dim.csv', 
                        usecols=['name', 'company_id'])
            .rename(columns={'name': 'company'}),
            pd.read_csv('csv_files_init/job_postings_fact.csv',
                        usecols=['job_id','company_id','job_title_short', 'salary_year_avg'])
            .query(f"job_title_short == '{role}'")
            .drop_duplicates(),
            on='company_id', how='inner')
        .merge(jobs, on='job_id', how='inner')
        .drop(columns=['skill_id', 'company_id', 'job_title_short'])
        .dropna(subset=['salary_year_avg'])
        .assign(skill_count=lambda x: x.groupby('job_id')['skills'].transform('count'))
        .assign(entry_count=lambda x: x.groupby('job_id')['skill_count'].transform('sum'))
        .assign(fraction=lambda x: x['skill_count'] / x['entry_count'])
        .sort_values(['entry_count', 'skills'], ascending=False)
        .drop(columns='skill_count')
        .drop_duplicates()
        .pivot(index=['job_id', 'company', 'entry_count', 'salary_year_avg'], columns='skills', values='fraction').reset_index().fillna(0)
        )
        

    df = df.sample(n=100, random_state=42) if (limit & (df.shape[0] > 100)) else df

    print(df.shape)
    print(df.head())

    # fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # sns.scatterplot(data=df, x='fraction', y='skill_count', hue='skills', alpha=0.5, ax=axes[0])
    # axes[0].set_title("Seaborn Scatter Plot by Group")
    # axes[0].set_xlabel("Fraction of Skills")
    # axes[0].set_ylabel("Skill Count")
    # axes[0].legend(title='Skills', bbox_to_anchor=(1.05, 1), loc='upper left')

    # sns.kdeplot(data=df, x='fraction', hue='skills', multiple='layer', ax=axes[1])
    # axes[1].set_title("Seaborn Histogram by Group")

    # plt.tight_layout()
    # plt.show()
    print(df.columns)
    # Example: df is your DataFrame with numeric columns
    # Remove non-numeric or identifier columns if necessary
    X = df.select_dtypes(include='number').drop(columns=['entry_count', 'salary_year_avg', 'job_id'])

    print(X.head())
    
    # Standardize
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=N_PCs)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_skill_order = pd.DataFrame(list(X.columns), columns=['skills'])
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(N_PCs)])

    # Example usage:
    for pc in range(0, N_PCs):
        print(f"PC{pc+1} explained variance: {pca.explained_variance_ratio_[pc]:.2f}")

    fig, axes = plt.subplots(nrows=1, ncols=N_PCs-1, figsize=(15, 3))
    for pc in range(0, N_PCs-1):
        ax = axes[pc]
        pc1 = f'PC{pc+1}'
        pc2 = f'PC{pc+2}'
        ax.scatter(pca_df[pc1], pca_df[pc2], alpha=0.5)
        ax.set_xlabel(pc1)
        ax.set_ylabel(pc2)

    plt.tight_layout()
    plt.show(block=False)

    reducer = umap.UMAP(n_neighbors = umap_n_neighbors, init='pca', min_dist=umap_min_dist)
    embedding = reducer.fit_transform(X_scaled)

    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    umap_df = pd.concat([umap_df, df.reset_index(drop=True)], axis=1)

    clustering = DBSCAN(eps=clst_eps, min_samples=clst_min_samples).fit(embedding)
    umap_df["cluster"] = clustering.labels_
    umap_df['cluster_size'] = umap_df.groupby('cluster')['cluster'].transform('count')
    
    # save UMAP embeddings
    if SAVE:
        df.to_csv('files/skills.csv', index=False)
        umap_df.to_csv('files/umap_results.csv', index=False)
        pca_df.to_csv('files/pca_results.csv', index=False)
        joblib.dump(pca, 'files/pca_model.pkl')
        joblib.dump(embedding, 'files/umap_model.pkl')
        pca_skill_order.to_csv('files/pca_skill_order.csv', index=False)
        
    return df, umap_df, pca, pca_skill_order
    

if __name__ == "__main__":
    main()
# %%
