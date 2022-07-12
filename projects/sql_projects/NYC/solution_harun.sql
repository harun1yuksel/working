CREATE DATABASE NYC_taxi

SELECT * into sep_oct
FROM
    (SELECT * from sep
    UNION
    SELECT * from oct) a --- bu sondaki alians kullanılmasa bile bu olmadan hata veriyor.

--- 1. Most expensive trip (total amount).

SELECT max(Total_amount)
from sep_oct 

---    2. Most expensive trip per mile (total amount/mile).

SELECT max( 1.0 * Total_amount / Trip_distance)
from sep_oct
WHERE Trip_distance != 0

---    3. Most generous trip (highest tip).

SELECT max(Tip_amount)
from sep_oct

SELECT * 
FROM sep_oct
WHERE Tip_amount = (select max(Tip_amount) from sep_oct)  --- or below

SELECT top 1 *
FROM sep_oct
order by Tip_amount DESC

---    4. Longest trip duration.

SELECT top 1 *, Datediff(minute, [lpep_pickup_datetime], [Lpep_dropoff_datetime]) as trip_duration
FROM sep_oct
Order By (trip_duration) DESC

 




