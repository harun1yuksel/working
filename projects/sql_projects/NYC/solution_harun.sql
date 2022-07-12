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

SELECT top 1 Datediff(minute, [lpep_pickup_datetime], [Lpep_dropoff_datetime]) as trip_duration, * 
FROM sep_oct
Order By (trip_duration) DESC

---    5. Mean tip by hour.

SELECT  distinct datepart(HOUR, [lpep_pickup_datetime]) as hour_in_day, 
round(avg(Tip_amount) over(partition by datepart(HOUR, [lpep_pickup_datetime])),2,2) as avg_tip
FROM sep_oct
Order By 1
 
---     6. Median trip cost (This question is optional. You can search for “median” calculation if you want).

SELECT distinct PERCENTILE_DISC(0.5) WITHIN GROUP(Order by Total_amount) over() as median_amount
from sep_oct

/*    7. Average total trip by day of week (Fortunately, we have day of week information. 
Otherwise, we need to create a new date column without hours from date column. 
Then, we need to create "day of week" column, i.e Monday, Tuesday .. or 1, 2 ..,  from that new date column. 
Total trip count should be found for each day, lastly average total trip should be calculated for each day). */

SELECT distinct lpep_pickup_day_of_week, count(*) over(PARTITION by lpep_pickup_day_of_week) as trip_count
from sep_oct

SELECT distinct datepart(day,lpep_pickup_datetime) as day_of_month, lpep_pickup_day_of_week, 
count(*) over(PARTITION by datepart(day,lpep_pickup_datetime) ) as trip_count
from sep_oct

with A as (
    SELECT distinct datepart(day,lpep_pickup_datetime) as day_of_month, lpep_pickup_day_of_week, 
    count(*) over(PARTITION by datepart(day,lpep_pickup_datetime) ) as trip_count
    from sep_oct
)
SELECT distinct lpep_pickup_day_of_week, avg(trip_count) over (PARTITION by lpep_pickup_day_of_week) avg_trip_per_day
From A 
