CREATE DATABASE conversionDB

CREATE TABLE conversions(
	Visitor_ID BIGINT primary key,
	Adv_Type VARCHAR(20),
	visitor_ACTION VARCHAR(20)
    )

    INSERT conversions (Visitor_ID, Adv_Type, visitor_ACTION)
    VALUES 
            (1, 'A', 'Left'),
            (2, 'A', 'Order'),
            (3, 'B', 'Left'),
            (4, 'A', 'Order'),
            (5, 'A', 'Review'),
            (6, 'A', 'Left'),
            (7, 'B', 'Left'),
            (8, 'B', 'Order'),
            (9, 'B', 'Review'),
            (10, 'A', 'Review')
   GO


Create VIEW conversion_count AS
SELECT Adv_Type, COUNT(visitor_ACTION)
                 as Conversion_Rate
FROM conversions A
WHERE visitor_ACTION = 'order'
GROUP BY Adv_Type

SELECT *
from conversion_count

SELECT Adv_Type, COUNT(Adv_Type) as ADV_count
FROM conversions
GROUP By Adv_Type

CREATE VIEW num_ADV as
SELECT Adv_Type, COUNT(Adv_Type) as ADV_count
FROM conversions
GROUP By Adv_Type

SELECT *
from conversion_count

SELECT * 
from num_ADV

SELECT A.Adv_Type, conversion_rate, Adv_count
FROM conversion_count A, num_ADV B
WHERE A.Adv_Type = B.Adv_Type

create VIEW conver AS
SELECT A.Adv_Type, conversion_rate, Adv_count
FROM conversion_count A, num_ADV B
WHERE A.Adv_Type = B.Adv_Type

SELECT Adv_Type, cast(conversion_rate * 1.0 / Adv_count  as decimal (3,2)) as Conversion_Rate
from CONVER

