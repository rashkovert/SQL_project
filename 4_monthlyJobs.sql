-- Creating January Jobs table
CREATE TABLE january_jobs AS
    SELECT *
    FROM job_postings_fact
    WHERE EXTRACT(MONTH FROM job_posted_date) = 1;

-- Creating February Jobs table
CREATE TABLE february_jobs AS
    SELECT *
    FROM job_postings_fact
    WHERE EXTRACT(MONTH FROM job_posted_date) = 2;

-- Creating March Jobs table
CREATE TABLE march_jobs AS
    SELECT *
    FROM job_postings_fact
    WHERE EXTRACT(MONTH FROM job_posted_date) = 3;

