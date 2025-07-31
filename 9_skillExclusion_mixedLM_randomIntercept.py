# %%
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import mixedlm

SAVE = True
SKILL_COUNT = 0

def main():
    global df, df_tests, results_mixlm
    df, skills = make_df(skill_count=SKILL_COUNT)
    df_list, skills_list = make_df_list(df, skills)
    terms = {'excel', 'word', 'powerpoint'}
    common_set_inclusive = valid_exclusion_groupings(skills_list, terms)

    df_tests = run_t_tests(df_list, common_set_inclusive, terms)
    #plot_t_tests(df_tests, terms)
    hist_means_differences(df_tests, terms)
    
    results_mixlm, model_mixlm, df_mixlm = mixlm_make(df_tests)
    mixlm_plot_effect_ci(results_mixlm, df_mixlm, terms)

    mixlm_effects(results_mixlm, df_mixlm)

    input("Press Enter to exit...")
    
    
    

def make_df(skill_count=0):
    skills = pd.read_csv('files/jobSkills_long.csv').query(f"count > {skill_count}")['skills']

    jobs = pd.merge(
        pd.read_csv('csv_files_init/skills_dim.csv', usecols=['skill_id', 'skills'])
            .query('skills in @skills'),
        pd.read_csv('csv_files_init/skills_job_dim.csv', usecols=['job_id', 'skill_id']),
        on='skill_id', how='inner')

    df = (
        pd.merge(
            pd.read_csv('csv_files_init/company_dim.csv', usecols=['name', 'company_id'])
                .rename(columns={'name': 'company'}),
            pd.read_csv('csv_files_init/job_postings_fact.csv',
                        usecols=['job_id','company_id','job_title_short', 'salary_year_avg'])
                .query("job_title_short == 'Data Analyst'")
                .drop_duplicates(),
            on='company_id', how='inner')
        .merge(jobs, on='job_id', how='inner')
        .drop(columns=['skill_id', 'company_id', 'job_title_short'])
        .dropna(subset=['salary_year_avg'])
        .drop_duplicates()
        .assign(exists=1)
        .pivot(index=['job_id', 'company', 'salary_year_avg'], columns='skills', values='exists')
        .reset_index()
        .fillna(0)
    )
    return df, skills

def make_df_list(df, skills, min_listings=0):
    skills_df = df[skills].copy()
    skills_df['skills_list'] = (
        skills_df.astype(bool)
                 .apply(lambda row: skills_df.columns[row].tolist(), axis=1)
    )
    df_list = pd.concat([
        df[['company', 'salary_year_avg', 'job_id']],
        skills_df.drop(columns=skills)
    ], axis=1)

    skills_list = df_list['skills_list'].value_counts()[lambda x: x > min_listings].index.tolist()
    return df_list, skills_list

def lists_includeExclude(skills_list, terms):
    matches = [entry for entry in skills_list if any(term in entry for term in terms)]
    match_omit = [[skill for skill in sublist if skill not in terms] for sublist in matches]
    misses = [entry for entry in skills_list if all(term not in entry for term in terms)]
    return matches, match_omit, misses

def valid_exclusion_groupings(skills_list, terms):
    matches, match_omit, misses = lists_includeExclude(skills_list, terms)
    common_list_inclusive = [matches[i] for i, sublist in enumerate(match_omit) if sublist in misses]
    return set(tuple(sublist) for sublist in common_list_inclusive)

def run_t_tests(df_list, common_set_inclusive, terms):
    tests = []

    for terms_inclusive in sorted(common_set_inclusive):
        terms_exclusive = tuple(s for s in terms_inclusive if s not in terms)

        s_inc = df_list[df_list['skills_list'].apply(lambda l: tuple(l) == terms_inclusive)]['salary_year_avg'].dropna()
        s_exc = df_list[df_list['skills_list'].apply(lambda l: tuple(l) == terms_exclusive)]['salary_year_avg'].dropna()

        t_stat, p_val = ttest_ind(s_inc, s_exc, equal_var=False) # Welch's, assuming jobs with common credentials exhibit roughly gaussian salaries

        if not pd.isna(p_val):
            tests.append({
                'set_inclusive': terms_inclusive,
                'set_exclusive': terms_exclusive,
                'p_val': p_val,
                'salaries_inclusive': s_inc,
                'salaries_exclusive': s_exc,
                'mean_inclusive': np.mean(s_inc),
                'mean_exclusive': np.mean(s_exc)
            })

    df_tests = pd.DataFrame(tests)
    valid_pvals = df_tests['p_val'].dropna()
    rejected, pvals_corrected, _, _ = multipletests(valid_pvals, alpha=0.05, method='fdr_bh')

    df_tests.loc[valid_pvals.index, 'q_val'] = pvals_corrected
    df_tests.loc[valid_pvals.index, 'significant'] = rejected
    df_tests = df_tests.sort_values(by='q_val').reset_index(drop=True)
    
    return df_tests


def plot_t_tests(df_tests, terms):
    n = len(df_tests)
    ncols = int(np.ceil(n ** 0.5))+3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 1.3, 1 * nrows))
    axes = axes.flatten()
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    fig.suptitle("Salary differences under exclusion of: " + ', '.join(terms), fontsize=24)

    for row in df_tests.itertuples(index=True):
        plot_df = pd.concat([
            pd.DataFrame({'salary': row.salaries_inclusive, 'group': 'inclusive'}),
            pd.DataFrame({'salary': row.salaries_exclusive, 'group': 'exclusive'})
        ])

        ax = axes[row.Index]
        sns.boxplot(data=plot_df, x='group', y='salary', ax=ax)
        ax.set_title(f'{row.set_exclusive}\np={row.p_val:.2f}, q={row.q_val:.3g}', fontsize=4.5)
        ax.tick_params(axis='both', labelsize=4)
        ax.tick_params(axis='x', rotation=20)
        ax.set_xlabel(ax.get_xlabel(), fontsize=4)
        ax.set_ylabel(ax.get_ylabel(), fontsize=4)
        if row.significant:
            ax.set_title(ax.get_title() + ' ★', color='crimson', fontsize=4)

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if SAVE:
        plt.savefig(f"9_t_test.pdf", bbox_inches='tight')
    plt.show(block=False)
    
def hist_means_differences(df_tests, terms):
    
    data = pd.DataFrame({'differences':df_tests['mean_inclusive']-df_tests['mean_exclusive']})
    mean_val = round(np.mean(data['differences'])*1e-3, 1)*1e3
    
    plt.figure(figsize=(5, 5))
    
    sns.histplot(data=data, x='differences', bins=int(len(data)/4), kde=True)
    plt.title("Histogram of salary mean differences adding skills: " + ', '.join(terms))
    plt.xlabel("Mean difference under inclusion")
    plt.ylabel("Count")
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean = {mean_val}')
    plt.legend()
    
    if SAVE:
        plt.savefig(f"9_hist.pdf", bbox_inches='tight')
    plt.show(block=False)

def mixlm_make(df):
    df = (
        df[['set_exclusive', 'salaries_inclusive', 'salaries_exclusive']]
        .melt(id_vars='set_exclusive',
            value_vars=['salaries_inclusive', 'salaries_exclusive'],
            var_name='group',
            value_name='salary_list')
        .explode('salary_list', ignore_index=True)
        .rename(columns={'salary_list': 'salary', 'set_exclusive': 'subgroup'})
    )

    #df['subgroup'] = df.apply(
    #    lambda row: tuple(sorted(set(row['subgroup']) | {row['group'].split('_', 1)[1]})),
    #    axis=1
    #)
    df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
    df['group'] = df['group'].astype('category')

    model_mixlm = mixedlm("salary ~ group", df, groups=df["subgroup"])
    results_mixlm = model_mixlm.fit()
    print(results_mixlm.summary())
    return results_mixlm, model_mixlm, df

        
def mixlm_plot_effect_ci(mdf, df, terms):
    
    group_vals = df[['group']].drop_duplicates().reset_index(drop=True)
    group_vals['subgroup'] = [df['subgroup'].iloc[0]] * len(group_vals)

    # Predict using fixed effects only
    group_vals['predicted'] = mdf.predict(group_vals)

    # ---------------------------
    # Estimate standard error of predictions
    # ---------------------------

    # Get covariance matrix of fixed effect estimates
    fe_cov = mdf.cov_params()

    # Manually build the design matrix X used for prediction
    # If group is categorical, we need to dummy code it (like the model does)
    group_dummies = pd.get_dummies(group_vals['group'], drop_first=True)

    # Add intercept column (1s), and ensure columns match model's design
    X = pd.concat([pd.Series(1, name='Intercept', index=group_vals.index), group_dummies], axis=1)
    X = X.reindex(columns=mdf.model.exog_names, fill_value=0)  # Ensure correct column order

    # Compute standard errors for predicted values using:
    # SE = sqrt(X_i.T @ cov_matrix @ X_i)
    se = np.sqrt(np.sum(np.dot(X, fe_cov.iloc[0:2, 0:2]) * X, axis=1))

    # Add 95% confidence intervals
    group_vals["ci_lower"] = group_vals["predicted"] - 1.96 * se
    group_vals["ci_upper"] = group_vals["predicted"] + 1.96 * se
    
    p_val = mdf.pvalues.get('group[T.salaries_inclusive]', None)

    group_vals['group'] = group_vals['group'].str.split('_').str[1]

    # ---------------------------
    # Plot the predicted means with confidence intervals
    # ---------------------------

    # Plot predicted means as points
    sns.pointplot(data=group_vals, x='group', y='predicted', join=False, color='steelblue')

    # Plot error bars for 95% CI
    plt.errorbar(group_vals['group'], group_vals['predicted'],
                yerr=1.96 * se, fmt='none', color='black', capsize=5)

    # Decorate plot
    plt.ylabel("Predicted Salary")
    plt.xlabel("Group: skill-set retention of (any of) " + ', '.join(terms))
    plt.title(f"Fixed-effect Predicted Salary by Group (95% CI): p={p_val:.3g}")
    plt.tight_layout()
    plt.show(block=False)
    
    
    plt.text(x=0.5, y=max(group_vals['predicted']) * 0.9,
         s=f'p = {p_val:.3g}' if p_val is not None else 'p = NA',
         fontsize=10, ha='left')
    
    if SAVE:
        plt.savefig(f"9_lm_diff_wFEvar.pdf", bbox_inches='tight')

def mixlm_effects(result, df):
    random_effects = result.random_effects  # dict: {group_name: effects}

    re_df = pd.DataFrame(random_effects) 
    re_df.columns = [
        tuple(skill for skill in col if pd.notna(skill))
        for col in re_df.columns]
    
    re_df = re_df.T

    print(re_df.head())
    
    plt.figure(figsize=(8,4))
    re_df['Group'].plot(kind='bar')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Random Intercepts by Group')
    plt.ylabel('Deviation from population intercept')
    plt.xticks(fontsize=4)
    plt.tight_layout()
    plt.show(block=False)

    random_slopes = re_df.get('group[T.salaries_inclusive]', None)
    if random_slopes is not None:
        slope_df = random_slopes.reset_index()
        slope_df.columns = ['Group', 'Random Slope']

        sns.histplot(slope_df['Random Slope'], kde=True)
        plt.title('Distribution of Random Slopes for group[T.salaries_inclusive]')
        plt.xlabel('Deviation from Fixed Effect')
        plt.show(block=False)

if __name__ == "__main__":
    main()


# %%
