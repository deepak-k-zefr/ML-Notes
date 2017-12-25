

**VARCHAR(30)** 

Characters with an expected max length of 30

**NOT NULL** 

 Must contain a value

**NULL** 

 Doesn't require a value

**CHAR(2)** 

 Contains exactly 2 characters

**DEFAULT "PA"** 

 Receives a default value of PA

**MEDIUMINT** 

 Value no greater then 8,388,608

**UNSIGNED** 

 Can't contain a negative value

**DATE** 

 Stores a date in the format YYYY-MM-DD

**ENUM('M', 'F')** 

 Can contain either a M or F

**TIMESTAMP** 

 Stores date and time in this format YYYY-MM-DD-HH-MM-SS

**FLOAT** 

 A number with decimal spaces, with a value no bigger than 1.1E38 or smaller than -1.1E38

**INT** 

 Contains a number without decimals

**AUTO_INCREMENT** 

 Generates a number automatically that is one greater then the previous row

**AUTO_INCREMENT** 

 Generates a number automatically that is one greater then the previous row

**AUTO_INCREMENT** : 

Generates a number automatically that is one greater then the previous row


**AUTO_INCREMENT** :

Generates a number automatically that is one greater then the previous row

**PRIMARY KEY (SLIDE)**

Unique ID that is assigned to this row of data

	I. Uniquely identifies a row or record 
	
	II. Each Primary Key must be unique to the row 
	
	III. Must be given a value when the row is created and that value cannot be NULL 
	
	IV. The original value cannot be changed It should be short 
	
	V. Itâlics probably best to auto increment the value of the key



## Atomic Data & Table Templating ##

As your database increases in size, you are going to want everything to be organized, so that it can perform your queries quickly. If your tables are set up properly, your database will be able to crank through hundreds of thousands of bits of data in seconds.

**How do you know how to best set up your tables though? Just follow some simple rules:**

1. Every table should focus on describing just one thing. Ex. Customer Table would have name, age, location, contact information. It shouldnt contain lists of anything such as interests, job history, past address, products purchased, etc.
After you decide what one thing your table will describe, then decide what things you need to describe that thing. Refer to the customer example given in the last step.

2. Write out all the ways to describe the thing and if any of those things requires multiple inputs, pull them out and create a new table for them. For example, a list of past employers.

3. Once your table values have been broken down, we refer to these values as being atomic. Be careful not to break them down to a point in which the data is harder to work with. It might make sense to create a different variable for the house number, street name, apartment number, etc.; but by doing so you may make your self more work? That decision is up to you?

4. Some additional rules to help you make your data atomic: Dont have multiple columns with the same sort of information. Ex. If you wanted to include a employment history you should create job1, job2, job3 columns. Make a new table with that data instead.

5. Dont include multiple values in one cell. Ex. You shouldnt create a cell named jobs and then give it the value: McDonalds, Radio Shack, Walmart.


**What does normalized mean?**
Normalized Tables-Normalized just means that the database is organized in a way that is considered standardized by professional SQL programmers. So if someone new needs to work with the tables they'll be able to understand how to easily.
Another benefit to normalizing your tables is that your queries will run much quicker and the chance your database will be corrupted will go down. 

**What are the rules for creating normalized tables:**

The tables and variables defined in them must be atomic Each row must have a Primary Key defined. Like your social security number identifies you, the Primary Key will identify your row. 

You also want to eliminate using the same values repeatedly in your columns. Ex. You wouldnt want a column named instructors, in which you hand typed in their names each time. You instead, should create an instructor table and link to its key. 

Every variable in a table should directly relate to the primary key. Ex. You should create tables for all of your customers potential states, cities and zip codes, instead of including them in the main customer table. Then you would link them using foreign keys. Note: Many people think this last rule is overkill and can be ignored! 

No two columns should have a relationship in which when one changes another must also change in the same table. This is called a Dependency. Note: This is another rule that is sometimes ignored.


------------ Numeric Types ------------

TINYINT: A number with a value no bigger than 127 or smaller than -128
SMALLINT: A number with a value no bigger than 32,768 or smaller than -32,767
MEDIUM INT: A number with a value no bigger than 8,388,608 or smaller than -8,388,608
INT: A number with a value no bigger than 2^31 or smaller than 2^31 1
BIGINT: A number with a value no bigger than 2^63 or smaller than 2^63 1
FLOAT: A number with decimal spaces, with a value no bigger than 1.1E38 or smaller than -1.1E38
DOUBLE: A number with decimal spaces, with a value no bigger than 1.7E308 or smaller than -1.7E308

------------ String Types ------------

CHAR: A character string with a fixed length
VARCHAR: A character string with a length thats variable
BLOB: Can contain 2^16 bytes of data
ENUM: A character string that has a limited number of total values, which you must define.
SET: A list of legal possible character strings. Unlike ENUM, a SET can contain multiple values in comparison to the one legal value with ENUM.

------------ Date & Time Types ------------

DATE: A date value with the format of (YYYY-MM-DD)
TIME: A time value with the format of (HH:MM:SS)
DATETIME: A time value with the format of (YYYY-MM-DD HH:MM:SS)
TIMESTAMP: A time value with the format of (YYYYMMDDHHMMSS)
YEAR: A year value with the format of (YYYY)

