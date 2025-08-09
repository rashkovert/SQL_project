# %%

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


# --- CONFIG ---
SKILL_COUNT = 0
MIN_ROLES = 3
CATEGORICAL_VARS = ["location"]
TARGET_VAR = "salary"
GROUP_VAR = "company"
SAVE=True

MIN_SHARE = 0.33

def main():
    df, skills = make_df(SKILL_COUNT)
        
    df_1 = df[df["company"].groupby(df["company"]).transform("count") >= MIN_ROLES]
    df_1, _ = make_df_list(df_1, skills)
    df_ols_1, model = fit_ols_with_fixed_effects(df_1)
    stats_df_1, results = compute_company_stats(df_ols_1)
    plot_significant_company_boxplots(df_ols_1, stats_df_1, use_residuals=True)
    
    if SAVE:
        stats_df_1.to_csv('files/10_company_salaries_controlled.csv', index=False)

    
    df_2, _ = make_df_list(df, skills)
    plot_top_company_barplots(df_2, use_residuals=True, top_share=MIN_SHARE)
    
    input("Press Enter to exit...")




def make_df(skill_count):
    
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
                        usecols=['job_id','company_id','job_title_short', 'salary_year_avg', 'job_location'])
            .query(f"job_title_short == 'Data Analyst'")
            .drop_duplicates(),
            on='company_id', how='inner')
        .merge(jobs, on='job_id', how='inner')
        .drop(columns=['skill_id', 'company_id', 'job_title_short'])
        .dropna(subset=['salary_year_avg'])
        .drop_duplicates()
        .assign(exists = 1)
        .pivot(index=['job_id', 'company', 'salary_year_avg', 'job_location'], columns='skills', values='exists').reset_index().fillna(0)
        .rename(columns={"salary_year_avg": "salary", "job_location": "location"})
    )
    
    
    return df, skills

def make_df_list(df, skills, min_listings=0):
    skills_df = df[skills].copy()
    skills_df['skills_list'] = (
        skills_df.astype(bool)
                 .apply(lambda row: skills_df.columns[row].tolist(), axis=1)
    )
    df_list = pd.concat([
        df[['company', 'salary', 'job_id', 'location']],
        skills_df.drop(columns=skills)
    ], axis=1)

    skills_list = df_list['skills_list'].value_counts()[lambda x: x > min_listings].index.tolist()
    
    df_list["skills"] = df_list["skills_list"].apply(lambda x: " ".join(x))
    
    return df_list, skills_list


def fit_ols_with_fixed_effects(df):
    formula = f"{TARGET_VAR} ~ " + " + ".join([f"C({var})" for var in CATEGORICAL_VARS])
    model = smf.ols(formula=formula, data=df).fit()
    df["residual"] = model.resid
    return df, model


def compute_company_stats(df, company_col='company', residual_col='residual', alpha=0.05):
    from scipy.stats import ttest_1samp

    results = []

    grouped = df[[company_col, residual_col, 'salary']].dropna().groupby(company_col)

    for company, group in grouped:
        residuals = group[residual_col]
        salaries = group['salary']
        
        mean_res = residuals.mean()
        t_stat, p_val = ttest_1samp(residuals, popmean=0, nan_policy='omit')
        
        results.append({
            company_col: company,
            'mean_salary': salaries.mean(),
            'mean_residual': mean_res,
            'p_value': p_val
        })

    stats_df = pd.DataFrame(results).set_index(company_col)
    
    # Apply BH correction
    reject, pvals_corrected, _, _ = multipletests(stats_df['p_value'], method='fdr_bh', alpha=alpha)
    stats_df['p_adj'] = pvals_corrected
    stats_df['significant'] = reject
    
    return stats_df, results

def plot_significant_company_boxplots(df, stats_df, company_col='company',
                                      residual_col='residual',
                                      use_residuals=True):
    
    filtered_df = df.merge(stats_df.query('significant'), on='company').sort_values('mean_residual')
    
    if use_residuals:
        val_col = residual_col
        filtered_df = filtered_df.sort_values('mean_residual')
    else:
        val_col = 'salary'
        filtered_df = filtered_df.sort_values('mean_salary')
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=filtered_df, x=val_col, y=company_col, palette='vlag')
    # sns.barplot(
    #     data=filtered_df,
    #     x=val_col,
    #     y=company_col,
    #     errorbar=('ci', 68)  # approx 1 SEM (68% confidence interval)
    # )
    
    if use_residuals:
        plt.axvline(0, linestyle='--', color='black')
        plt.xlabel("Salary Residual (Actual - Expected)")
    else:
        plt.xlabel("Salary")
        
    plt.yticks(fontsize=6)
    plt.ylabel("Company")
    plt.title(f"BH-Significant Over- and Under-Paying Companies\n(Regressed on location, >{MIN_ROLES-1} postings)")
    plt.tight_layout()
    
    if SAVE:
        plt.savefig("plots/10_SignificantSalaryingCompanies.pdf", bbox_inches='tight')
        
    plt.show(block=False)

def plot_top_company_barplots(df, top_share=0.5,
                              company_col='company', residual_col='residual',
                              use_residuals=False):
    
    top_companies = (
        df[company_col]
        .value_counts()
        .reset_index(name='count')
        .rename(columns={'index': 'company'})
        .assign(cumulative=lambda x: x['count'].cumsum() / x['count'].sum())
        .query('cumulative <= @top_share')['company']
    )

    filtered_df = df[df[company_col].isin(top_companies)]
    
    filtered_df, _ = fit_ols_with_fixed_effects(filtered_df)
    stats_df, _ = compute_company_stats(filtered_df)

    filtered_df = filtered_df.merge(stats_df.reset_index(), on='company', how='left')
    filtered_df['significant'] = filtered_df['significant'].fillna(False)

    if use_residuals:
        val_col = residual_col
        filtered_df = filtered_df.sort_values('mean_residual')
    else:
        val_col = 'salary'
        filtered_df = filtered_df.sort_values('mean_salary')
    

    plt.figure(figsize=(10, top_share*25))
    
    # sns.boxplot(
    #     data=,
    #     x=val_col, 
    #     y=company_col, 
    #     palette='vlag',
    #     hue='significant'
    # )
    
    sns.barplot(
        data=filtered_df,
        x=val_col,
        y=company_col,
        errorbar=('ci', 68),  # approx 1 SEM (68% confidence interval)
        hue='significant'
    )


    if use_residuals:
        plt.axvline(0, linestyle='--', color='black')
        plt.xlabel("Salary Residual (Actual - Expected)")
    else:
        plt.xlabel("Salary")

    plt.ylabel("Company")
    plt.title(f"Over- and Under-Paying Companies in Top {int(top_share * 100)}% of Postings (Regressed on location)")
    plt.legend(title="BH-Significant", loc='lower right')
    plt.yticks(fontsize=4)
    plt.tight_layout()

    if SAVE:
        plt.savefig("plots/10_TopSalaryingCompanies.pdf", bbox_inches='tight')

    plt.show(block=False)

    

if __name__ == "__main__":
    main()

# %%
