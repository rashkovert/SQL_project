/* 
PART 1:
Identify roles available in Boston, MA or Cambridge, MA by salary
*/

SELECT 
    job_title_short,
    salary_year_avg::INT,
    job_posted_date, 
    company_dim.name AS company_name
FROM job_postings_fact
LEFT JOIN company_dim ON 
    job_postings_fact.company_id = company_dim.company_id
WHERE 
    (job_location = 'Cambridge, MA' OR 
        job_location = 'Boston, MA') AND
    salary_year_avg IS NOT NULL
ORDER BY salary_year_avg ASC;


-- Identify average Boston/Cambridge salaries by company*title
SELECT 
    job_postings_fact.job_title_short,
    AVG(job_postings_fact.salary_year_avg)::INT AS mean_salary,
    company_dim.name AS company
FROM job_postings_fact
INNER JOIN
    company_dim ON job_postings_fact.company_id = company_dim.company_id
WHERE 
    (job_location = 'Cambridge, MA' OR 
        job_location = 'Boston, MA') AND
    job_postings_fact.salary_year_avg IS NOT NULL
GROUP BY job_postings_fact.job_title_short, company
ORDER BY mean_salary ASC;
-- saved as avgSalary_role_company.csv

-- Identify average Boston/Cambridge/NYC salaries by title
SELECT 
    job_title_short,
    AVG(salary_year_avg)::INT AS mean_salary,
    job_location
FROM job_postings_fact
WHERE 
    (job_location = 'Cambridge, MA' OR 
        job_location = 'Boston, MA' OR
        job_location = 'New York, NY') AND
    salary_year_avg IS NOT NULL
GROUP BY job_title_short, job_location
ORDER BY job_title_short ASC;
-- saved as avgSalary_role_location.csv

/* 
PART 2: skills required per job (Boston area)
ma_jobskills created
*/

SELECT 
    job_location,
    COUNT(*) AS job_count
FROM job_postings_fact
WHERE job_location LIKE '%, MA'
GROUP BY job_location
ORDER BY job_count DESC;
-- Boston, Cambridge, Waltham, Lexington, Worcester, Somerville, Watertown

CREATE TABLE MA_jobSkills AS
WITH jobs_MA AS (
    SELECT 
        job_id,
        job_title_short,
        salary_year_avg::INT
    FROM job_postings_fact
    LEFT JOIN company_dim ON 
        job_postings_fact.company_id = company_dim.company_id
    WHERE 
        job_location IN ('Boston, MA', 'Cambridge, MA', 
                        'Waltham, MA', 'Lexington, MA', 
                        'Worcester, MA', 'Somerville, MA', 
                        'Watertown, MA') AND
        salary_year_avg IS NOT NULL
    ORDER BY salary_year_avg ASC
)
SELECT 
    jobs_MA.job_title_short,
    skills_dim.skills,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY jobs_MA.job_title_short),
        2
    ) AS skill_percentage
FROM jobs_MA
INNER JOIN skills_job_dim ON skills_job_dim.job_id = jobs_MA.job_id
INNER JOIN skills_dim ON skills_dim.skill_id = skills_job_dim.skill_id
GROUP BY job_title_short, skills
ORDER BY job_title_short, skill_percentage DESC;
-- saved as TABLE: ma_jobskills, also skills_per_MAjob.csv 

/*
PART 2.5:
Role salaries per location (which is entry level?)
*/

-- JOB COUNTS (by location):
-- CREATE TABLE MA_jobCounts AS
WITH jobs_MA AS (
    SELECT 
        job_title_short,
        job_location,
        COUNT(*) AS count_per_location,
        AVG(salary_year_avg)::INT AS salary_avg
    FROM job_postings_fact
    WHERE 
        job_location IN ('Boston, MA', 'Cambridge, MA', 
                        'Waltham, MA', 'Lexington, MA', 
                        'Worcester, MA', 'Somerville, MA', 
                        'Watertown, MA') AND
        salary_year_avg IS NOT NULL
    GROUP BY job_title_short, job_location
    ORDER BY salary_avg DESC
)

SELECT 
    job_title_short,
    job_location,
    count_per_location,
    salary_avg,
    AVG(salary_avg) OVER (PARTITION BY job_title_short)::INT AS salaryAvgAvg
FROM jobs_MA
ORDER BY salaryAvgAvg DESC, salary_avg DESC;
--saved as TABLE: ma_jobcounts, also counts_per_MAjob.csv


/*
PART 3:
most in-demand skills for my role (Data Analyst)
*/
SELECT *
FROM ma_jobskills
WHERE job_title_short = 'Data Analyst';
-- Frequencies: sql: 15%, python: 12%, excel: 10%, tableau: 8



/*
PART 4:
top skills per job based on avg salary
*/

WITH jobs_MA AS (
    SELECT 
        job_title_short,
        COUNT(*) AS count_MA,
        AVG(salary_year_avg)::INT AS salary_avg
    FROM job_postings_fact
    WHERE 
        job_location IN ('Boston, MA', 'Cambridge, MA', 
                        'Waltham, MA', 'Lexington, MA', 
                        'Worcester, MA', 'Somerville, MA', 
                        'Watertown, MA') AND
        salary_year_avg IS NOT NULL
    GROUP BY job_title_short
    ORDER BY salary_avg DESC
), jobskills AS (
    SELECT job_title_short, skills, skill_percentage
    FROM (
        SELECT *, 
            ROW_NUMBER() OVER (PARTITION BY job_title_short ORDER BY skill_percentage DESC) AS rank
        FROM ma_jobskills
    ) ranked_skills
    WHERE rank <= 5
)

SELECT jobskills.*, 
    jobs_MA.count_ma, jobs_MA.salary_avg
FROM jobs_MA
RIGHT JOIN jobskills ON jobskills.job_title_short = jobs_MA.job_title_short
ORDER BY salary_avg DESC, skill_percentage DESC;

/*
PART 5: most optimal skills to learn
SAVING: jobSkills_long.csv
*/

SELECT 
    skills_dim.skills,
    job_title_short,
    COUNT(*),
    AVG(salary_year_avg)::INT AS avg_salary
    --company_dim.name AS company_name
FROM job_postings_fact
INNER JOIN skills_job_dim ON skills_job_dim.job_id = job_postings_fact.job_id
INNER JOIN skills_dim ON skills_dim.skill_id = skills_job_dim.skill_id
WHERE 
    job_title_short = 'Data Analyst' AND
    /*
    job_location IN ('Boston, MA', 'Cambridge, MA', 
                    'Waltham, MA', 'Lexington, MA', 
                    'Worcester, MA', 'Somerville, MA', 
                    'Watertown, MA') AND
    */
    salary_year_avg IS NOT NULL
GROUP BY skills, job_title_short
HAVING COUNT(*) > 10
ORDER BY avg_salary DESC;






