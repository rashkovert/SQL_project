CREATE DATABASE sql_course;

CREATE TABLE job_applied (
    job_id INT,
    application_sent_date DATE,
    custom_resume BOOLEAN,
    resume_file_name VARCHAR(255),
    cover_letter_sent BOOLEAN,
    cover_letter_file_name VARCHAR(255),
    status VARCHAR(50)
);

INSERT INTO job_applied (
            job_id,
            application_sent_date,
            custom_resume,
            resume_file_name,
            cover_letter_sent,
            cover_letter_file_name,
            status)
VALUES      (1, 
            '2024-02-01',
            true,
            'resume_01.pdf',
            true,
            'cover_letter_01.pdf',
            'submitted'), 

            (2, 
            '2024-02-02',
            false,
            'resume_02.pdf',
            true,
            'cover_letter_02.pdf',
            'submitted')

SELECT *
FROM job_applied;

-- add table
ALTER TABLE job_applied
ADD contact VARCHAR(50)

-- specify contents of new column
UPDATE  job_applied
SET     contact = 'Teddy Rashkover'
WHERE   job_id = 1;

UPDATE  job_applied
SET     contact = 'Nicole Rashkover'
WHERE   job_id = 2;

-- change column name
ALTER TABLE job_applied
RENAME COLUMN contact TO contact_name;

-- change column datatype
ALTER TABLE job_applied
ALTER COLUMN contact_name TYPE TEXT;

-- delete column
ALTER TABLE job_applied
DROP contact_name;

-- delete table
DROP TABLE job_applied;