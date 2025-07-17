import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

LOAD_CLUSTERS = True

def main():
    if not LOAD_CLUSTERS:
        print("Computing clusters for Data Analyst role...")
        umap_df, pca_df = compute_clusters(role='Data Analyst', skill_count=100, limit=0, N_PCs=10, clst_min_samples=25, umap_n_neighbors=50, SAVE=True)
    else:
        umap_df = pd.read_csv('files/umap_results.csv')
        umap_df['cluster_size'] = umap_df.groupby('cluster')['cluster'].transform('count')
        umap_df.to_csv('files/umap_results.csv', index=False)
        pca_df = pd.read_csv('files/pca_results.csv')
    plot_clusters(umap_df)
    plot_cluster_skills(umap_df, n_clusters=16)
    
def plot_cluster_skills(umap_df, n_clusters=16, div=None):
    if div is None: div = np.ceil(n_clusters**0.5).astype(int)
    # Show violin plots of skill frequencies for each cluster
    
    top_clusters = (
        umap_df[umap_df.cluster != -1][['cluster', 'cluster_size']]
        .drop_duplicates()
        .sort_values('cluster_size', ascending=False)
        .head(n_clusters)
    )
    
    fig, axes = plt.subplots(nrows=np.ceil(n_clusters/div).astype(int), ncols=div, figsize=(60/div, 2*np.ceil(n_clusters/div)), sharex=True)
    axes = axes.flatten() if n_clusters > div else [axes]
    for c in range(len(top_clusters['cluster'])):
        cluster = top_clusters['cluster'].iloc[c]
        cluster_size = top_clusters['cluster_size'].iloc[c]
        cluster_data = (
            umap_df[umap_df['cluster'] == cluster]
            .drop(columns=['UMAP1','UMAP2','company','entry_count','cluster','cluster_size']))
        cluster_data = cluster_data.melt(var_name='skills', value_name='fraction')
        sns.violinplot(data=cluster_data, x='skills', y='fraction', ax=axes[c], scale='width', inner='quartile')
        axes[c].set_title(f'Cluster {cluster}, Companies: {cluster_size}')
        axes[c].set_xlabel('Skills')
        axes[c].set_ylabel('Fraction of Skills')
        axes[c].tick_params(axis='x', rotation=90, labelbottom=True)
        
    for ax in axes[len(top_clusters):]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()
        
    

def compute_clusters(role='Data Analyst', skill_count = 200, limit = 0, N_PCs = 10, clst_min_samples = 50, umap_n_neighbors = 50, SAVE=True):
    
    skills = (pd.read_csv('files/jobSkills_long.csv').query(f"count > {skill_count}"))['skills']

    # Extract associated job IDs for these skills
    jobs = pd.merge(
        pd.read_csv('csv_files_init/skills_dim.csv', 
                    usecols=['skill_id', 'skills'])
                    .query('skills in @skills'),
        pd.read_csv('csv_files_init/skills_job_dim.csv', 
                    usecols=['job_id', 'skill_id']), 
        on='skill_id', how='inner')

    # Load company data
    df = (
        pd.merge(
            pd.read_csv('csv_files_init/company_dim.csv', 
                        usecols=['name', 'company_id'])
            .rename(columns={'name': 'company'}),
            pd.read_csv('csv_files_init/job_postings_fact.csv',
                        usecols=['job_id','company_id','job_title_short'])
            .query(f"job_title_short == '{role}'")
            .drop_duplicates(),
            on='company_id', how='inner')
        .merge(jobs, on='job_id', how='inner')
        .drop(columns=['job_id', 'skill_id', 'company_id', 'job_title_short'])
        .groupby(['company', 'skills']).size().reset_index(name='skill_count')
        .assign(entry_count=lambda x: x.groupby('company')['skill_count'].transform('sum'))
        .assign(fraction=lambda x: x['skill_count'] / x['entry_count'])
        .sort_values(['entry_count', 'skills'], ascending=False)
        .drop(columns='skill_count')
        .pivot(index=['company', 'entry_count'], columns='skills', values='fraction').reset_index().fillna(0)
        )

    df = df.sample(n=5000, random_state=42) if (limit & df.shape[0] > 5000) else df

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
    X = df.select_dtypes(include='number').drop(columns=['entry_count'])

    # Standardize
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=N_PCs)
    X_pca = pca.fit_transform(X_scaled)

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

    reducer = umap.UMAP(n_neighbors = umap_n_neighbors, init='pca')
    embedding = reducer.fit_transform(X_scaled)

    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    umap_df = pd.concat([umap_df, df.reset_index(drop=True)], axis=1)

    clustering = DBSCAN(eps=0.5, min_samples=clst_min_samples).fit(embedding)
    umap_df["cluster"] = clustering.labels_
    umap_df['cluster_size'] = umap_df.groupby('cluster')['cluster'].transform('count')
    
    # save UMAP embeddings
    if SAVE:
        umap_df.to_csv('files/umap_results.csv', index=False)
        pca_df.to_csv('files/pca_results.csv', index=False)
    return umap_df, pca_df
    
def plot_clusters(umap_df):
    fig2 = plt.figure(figsize=(10, 8))
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
        
    plt.show(block=False)
    
    

if __name__ == "__main__":
    main()