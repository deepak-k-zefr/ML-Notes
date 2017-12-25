
CREDITS: Derek Banas-https://www.youtube.com/watch?v=yPu6qV5byu4


## MySQL Tutorial A-Z

 **Logging in to MySQL**
 
 	 mysql5 -u mysqladmin -p
 
 **Quit**

	 Quit MySQL
   
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

**Add a max score column to test**

	ALTER TABLE test ADD maxscore INT NOT NULL AFTER type; 
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
