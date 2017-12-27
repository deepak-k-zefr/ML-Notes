
CREDITS: Derek Banas-https://www.youtube.com/watch?v=yPu6qV5byu4


## MySQL Tutorial A-Z

 **Logging in to MySQL**
 
 	 mysql -u root -p
 
 **Quit**
 		
		exit
	
**Display all databases**
   
    show databases;
    
    
**Create a database**
    
    CREATE DATABASE test2
    
**Make test2 the active database**
  
    USE test2
   
 **Show the currently selected database**
    
    SELECT DATABASE()
          
**Delete the  database**
    
    DROP DATABASE IF EXISTS test2
    
    
**Add Table in Database**
```
CREATE TABLE student(
first_name VARCHAR(30) NOT NULL,
last_name VARCHAR(30) NOT NULL,
email VARCHAR(60) NULL,
street VARCHAR(50) NOT NULL,
city VARCHAR(40) NOT NULL,
state CHAR(2) NOT NULL DEFAULT "PA",
zip MEDIUMINT UNSIGNED NOT NULL,
phone VARCHAR(20) NOT NULL,
birth_date DATE NOT NULL,
sex ENUM('M', 'F') NOT NULL,
date_entered TIMESTAMP,
lunch_cost FLOAT NULL,
student_id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY
);
```


**Show the table set up**

	DESCRIBE student
	
**Inserting Data into a Table**

```
INSERT INTO student VALUES('Harry', 'Truman', 'htruman@aol.com', 
	'202 South St', 'Vancouver', 'WA', 98660, '792-223-9810', "1946-1-24",
	'M', NOW(), 3.50, NULL);
	
	INSERT INTO student VALUES('Shelly', 'Johnson', 'sjohnson@aol.com', 
	'9 Pond Rd', 'Sparks', 'NV', 89431, '792-223-6734', "1970-12-12",
	'F', NOW(), 3.50, NULL);
	
	INSERT INTO student VALUES('Bobby', 'Briggs', 'bbriggs@aol.com', 
	'14 12th St', 'San Diego', 'CA', 92101, '792-223-6178', "1967-5-24",
	'M', NOW(), 3.50, NULL);
	
	INSERT INTO student VALUES('Donna', 'Hayward', 'dhayward@aol.com', 
	'120 16th St', 'Davenport', 'IA', 52801, '792-223-2001', "1970-3-24",
	'F', NOW(), 3.50, NULL);
	
	INSERT INTO student VALUES('Audrey', 'Horne', 'ahorne@aol.com', 
	'342 19th St', 'Detroit', 'MI', 48222, '792-223-2001', "1965-2-1",
	'F', NOW(), 3.50, NULL);
	
	INSERT INTO student VALUES('James', 'Hurley', 'jhurley@aol.com', 
	'2578 Cliff St', 'Queens', 'NY', 11427, '792-223-1890', "1967-1-2",
	'M', NOW(), 3.50, NULL);
	
	INSERT INTO student VALUES('Lucy', 'Moran', 'lmoran@aol.com', 
	'178 Dover St', 'Hollywood', 'CA', 90078, '792-223-9678', "1954-11-27",
	'F', NOW(), 3.50, NULL);
	
	INSERT INTO student VALUES('Tommy', 'Hill', 'thill@aol.com', 
	'672 High Plains', 'Tucson', 'AZ', 85701, '792-223-1115', "1951-12-21",
	'M', NOW(), 3.50, NULL);
	
	INSERT INTO student VALUES('Andy', 'Brennan', 'abrennan@aol.com', 
	'281 4th St', 'Jacksonville', 'NC', 28540, '792-223-8902', "1960-12-27",
	'M', NOW(), 3.50, NULL);
```

**Show all student Data**

		SELECT * from student
		
		
**Create a Table for classes**

``` 
	CREATE TABLE class(
	name VARCHAR(30) NOT NULL,
	class_id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY);
```
	
**Show all Tables**

	SHOW tables
	
**Insert all possible classes**

``` 
INSERT INTO class VALUES
('English', NULL), ('Speech', NULL), ('Literature', NULL),
('Algebra', NULL), ('Geometry', NULL), ('Trigonometry', NULL),
('Calculus', NULL), ('Earth Science', NULL), ('Biology', NULL),
('Chemistry', NULL), ('Physics', NULL), ('History', NULL),
('Art', NULL), ('Gym', NULL);
```

**Create Tables test,score and absence**


![SCHEMA](schema.png?raw=true)


```
CREATE TABLE test(
	date DATE NOT NULL,
	type ENUM('T', 'Q') NOT NULL,
	class_id INT UNSIGNED NOT NULL,
	test_id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY);
```	
```
 CREATE TABLE score(
	student_id INT UNSIGNED NOT NULL,
	event_id INT UNSIGNED NOT NULL,
	score INT NOT NULL,
	PRIMARY KEY(event_id, student_id));
```

```
CREATE TABLE absence(
	student_id INT UNSIGNED NOT NULL,
	date DATE NOT NULL,
	PRIMARY KEY(student_id, date));
```



We combined the event and student id to make sure we don't have 
duplicate scores and it makes it easier to change scores

Since neither the event or the student ids are unique on their 
own we are able to make them unique by combining them.


Again we combine 2 items that aren't unique to generate a 
unique key.

**ADD a new column(max score column to test)**
	ALTER TABLE test ADD maxscore INT NOT NULL AFTER type; 
	DESCRIBE test;

**Change the name of a column(event_id in score to test_id)**

	ALTER TABLE score CHANGE event_id test_id 
	INT UNSIGNED NOT NULL;
	DESCRIBE test;

	
**Insert Tests**

``` 	
	INSERT INTO test VALUES
	('2014-8-25', 'Q', 15, 1, NULL),
	('2014-8-27', 'Q', 15, 1, NULL),
	('2014-8-29', 'T', 30, 1, NULL),
	('2014-8-29', 'T', 30, 2, NULL),
	('2014-8-27', 'Q', 15, 4, NULL),
	('2014-8-29', 'T', 30, 4, NULL);
	
```

	SELECT * FROM test;




**Enter student scores**

```
INSERT INTO score VALUES
	(1, 1, 15),
	(1, 2, 14),
	(1, 3, 28),
	(1, 4, 29),
	(1, 5, 15),
	(1, 6, 27),
	(2, 1, 15),
	(2, 2, 14),
	(2, 3, 26),
	(2, 4, 28),
	(2, 5, 14),
	(2, 6, 26),
	(3, 1, 14),
	(3, 2, 14),
	(3, 3, 26),
	(3, 4, 26),
	(3, 5, 13),
	(3, 6, 26),
	(4, 1, 15),
	(4, 2, 14),
	(4, 3, 27),
	(4, 4, 27),
	(4, 5, 15),
	(4, 6, 27),
	(5, 1, 14),
	(5, 2, 13),
	(5, 3, 26),
	(5, 4, 27),
	(5, 5, 13),
	(5, 6, 27),
	(6, 1, 13),
	(6, 2, 13),
	# Missed this day (6, 3, 24),
	(6, 4, 26),
	(6, 5, 13),
	(6, 6, 26),
	(7, 1, 13),
	(7, 2, 13),
	(7, 3, 25),
	(7, 4, 27),
	(7, 5, 13),
	# Missed this day (7, 6, 27),
	(8, 1, 14),
	# Missed this day (8, 2, 13),
	(8, 3, 26),
	(8, 4, 23),
	(8, 5, 12),
	(8, 6, 24),
	(9, 1, 15),
	(9, 2, 13),
	(9, 3, 28),
	(9, 4, 27),
	(9, 5, 14),
	(9, 6, 27),
	(10, 1, 15),
	(10, 2, 13),
	(10, 3, 26),
	(10, 4, 27),
	(10, 5, 12),
	(10, 6, 22);
```

**Fill absences Table**

```
INSERT INTO absence VALUES
	(6, '2014-08-29'),
	(7, '2014-08-29'),
	(8, '2014-08-27');
```

Now we are done filling all the data.

**Select specific columns from a table**

```
SELECT FIRST_NAME, last_name 
	FROM student;
```

**Rename Tables**

```
	RENAME TABLE 
	absence to absences,
	class to classes,
	score to scores,
	student to students,
	test to tests;
```

### USE WHERE- 
**Show every student born in the state of Washington**
```
	SELECT first_name, last_name, state 
	FROM students
	WHERE state="WA";
```
**Show every student born after 1965**

```
	SELECT *
	FROM students
	WHERE YEAR(birth_date) >= 1965;
```
	a. You can compare values with =, >, <, >=, <=, !=
	
	b. To get the month, day or year of a date use MONTH(), DAY(), or YEAR()

**Show every student born in February or California**

```
. 	SELECT *
	FROM students
	WHERE MONTH(birth_date) = 2 OR state="CA";
```

	a. AND, && : Returns a true value if both conditions are true 

	b. OR, || : Returns a true value if either condition is true 

	c. NOT, ! : Returns a true value if the operand is false

**Show every student born in February AND (California or Nevada)**
```	
	SELECT **
	FROM students
	WHERE DAY(birth_date) >= 12 && (state="CA" || state="NV");
```

**Return rows that have a specific(last_name) empty value**

```	SELECT *
	FROM students
	WHERE last_name IS NULL;
```

**Sort results by a specific(last_name) column.**

```
	SELECT *
	FROM students
	ORDER BY last_name;

ADD ASC or DESC to specify order
```

## LIMIT

**Show first 5 results**

```	SELECT *
	FROM students
	LIMIT 5;
```
**Show 5-10 results**

```	SELECT *
	FROM students
	LIMIT 5, 10;
```

## CONCAT

**Concat first name and last name**

```	SELECT CONCAT(first_name, " ", last_name) AS 'Name',
	CONCAT(city, ", ", state) AS 'Hometown'
	FROM students;
```
	a. CONCAT is used to combine results
	b. AS provides for a way to define the column name
	

**Match any first name that starts with a D, or ends with a n**

``` 
	SELECT last_name, first_name
	FROM students
	WHERE first_name LIKE 'D%' OR last_name LIKE '%n';
```

**MATCH _ _ _ Y last names**

``` 
	SELECT last_name, first_name
	FROM students
	WHERE first_name LIKE '___y';
```
**Show all the categories of a column(state)**

```
	SELECT DISTINCT state
	FROM students
	ORDER BY state;
```

**Show count of all the categories of a column(state)**
```	
	SELECT COUNT(DISTINCT state)
	FROM students;
```

**Show count matching a condition**
```
	SELECT COUNT(*)
	FROM students
	WHERE sex='M';
```

**Group results based on a category(sex/birth year)**
```
 SELECT sex, COUNT(*)
	FROM students
	GROUP BY sex;
```

```SELECT MONTH(birth_date) AS 'Month', COUNT(*)
	FROM students
	GROUP BY Month
	ORDER BY Month;
```

``` SELECT state, COUNT(state) AS 'Amount'
	FROM students
	GROUP BY state
	HAVING Amount > 1;
```
	a. HAVING allows you to narrow the results after the query is executed


**Select based on a condition**
	
	SELECT student_id, test_id
	FROM scores
	WHERE student_id = 6;

**Insert into table**
	
	INSERT INTO scores VALUES
	(6, 3, 24);

**Delete based on a condition**
	
	DELETE FROM absences 
	WHERE student_id = 6;

**ADD COLUMN**
	
	ALTER TABLE absences
	ADD COLUMN test_taken CHAR(1) NOT NULL DEFAULT 'F'
	AFTER student_id; 
	
	
Use ALTER to add a column to a table. You can use AFTER
or BEFORE to define the placement

**Update a value based on a condition**

	UPDATE scores SET score=25 
	WHERE student_id=4 AND test_id=3;

**Use BETWEEN to find matches between a minimum and maximum**

	SELECT first_name, last_name, birth_date
	FROM students
	WHERE birth_date 
	BETWEEN '1960-1-1' AND '1970-1-1';

**Use IN to narrow results based on a predefined list of options**
	
	SELECT first_name, last_name
	FROM students
	WHERE first_name IN ('Bobby', 'Lucy', 'Andy');

**JOIN- Get info from multiple sources:**


	SELECT student_id, date, score, maxscore
	FROM tests, scores
	WHERE date = '2014-08-25'
	AND tests.test_id = scores.test_id;
	


a. To combine data from multiple tables you can perform a JOIN
by matching up common data like we did here with the test ids

b. You have to define the 2 tables to join after FROM

c. You have to define the common data between the tables after WHERE

	SELECT scores.student_id, tests.date, scores.score, tests.maxscore
	FROM tests, scores
	WHERE date = '2014-08-25'
	AND tests.test_id = scores.test_id;


**JOIN + GROUPBY**

	SELECT students.student_id, 
	CONCAT(students.first_name, " ", students.last_name) AS Name,
	COUNT(absences.date) AS Absences
	FROM students, absences
	WHERE students.student_id = absences.student_id
	GROUP BY students.student_id;
	
	SELECT students.student_id, 
	CONCAT(students.first_name, " ", students.last_name) AS Name,
	COUNT(absences.date) AS Absences
	FROM students LEFT JOIN absences
	ON students.student_id = absences.student_id
	GROUP BY students.student_id;

If we need to include all information from the table listed
first "FROM students", even if it doesn't exist in the table on
the right "LEFT JOIN absences", we can use a LEFT JOIN.

	SELECT students.first_name, 
	students.last_name,
	scores.test_id,
	scores.score
	FROM students
	INNER JOIN scores
	ON students.student_id=scores.student_id
	WHERE scores.score <= 15
	ORDER BY scores.test_id;

a. An INNER JOIN gets all rows of data from both tables if there
is a match between columns in both tables

b. Here I'm getting all the data for all quizzes and matching that 
data up based on student ids
	
