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

---    8. Count of trips by hour (Luckily, we have hour column. Otherwise, a new hour column should be created from date column, then count trips by hour).
SELECT *
from sep_oct

SELECT  distinct datepart(HOUR, [lpep_pickup_datetime]) as hour_in_day, 
COUNT(*) over(partition by datepart(HOUR, [lpep_pickup_datetime])) as trip_count
from sep_oct

---    9. Average passenger count per trip.

SELECT cast((1.0 * sum(Passenger_count) / COUNT(*)) as decimal (4, 2))
from sep_oct

select cast(avg(1.0 * Passenger_count) as decimal (4 , 2))
from sep_oct

----   10. Average passenger count per trip by hour.

SELECT  distinct datepart(HOUR, [lpep_pickup_datetime]) as hour_in_day,
cast((avg(1.0 * Passenger_count) over(PARTITION by datepart(HOUR, lpep_pickup_datetime))) as decimal (4, 2)) as trip_count_per_hour
from sep_oct

---    11. Which airport welcomes more passengers: JFK or Newark? Tip: check RateCodeID from data dictionary for the definition (2: JFK, 3: Newark).

SELECT RateCodeID, sum(Passenger_count)
from sep_oct
group by RateCodeID
ORDER by 1


select distinct
    case when RateCodeID = '1' then 'Standart Rate'
        when RateCodeID = '2' then 'JFK'
        when RateCodeID = '3' then 'Newark'
        when RateCodeID = '4' then 'Nassau or Westchester'
        when RateCodeID = '5' then 'Negoatiated Fare'
        when RateCodeID = '6' then 'Group Ride'
    end as RateCodeID,
    sum(Passenger_count) OVER(PARTITION by RateCodeID) as passenger_count
from sep_oct
ORDER by 2

         ---- or

select distinct
    case when RateCodeID = '1' then 'Standart Rate'
        when RateCodeID = '2' then 'JFK'
        when RateCodeID = '3' then 'Newark'
        when RateCodeID = '4' then 'Nassau or Westchester'
        when RateCodeID = '5' then 'Negoatiated Fare'
        when RateCodeID = '6' then 'Group Ride'
    end as RateCodeID,
sum(Passenger_count) as passenger_count
from sep_oct
GROUP by RateCodeID
ORDER by 2

---   12. How many nulls are there in Total_amount?

select sum(case when Total_amount is null then 1 else 0 end) as count_nulls,
       count(Total_amount) count_not_nulls 
from sep_oct

  ---- 13. How many values are there in Trip_distance? (count of non-missing values)

  SELECT Trip_distance
  from sep_oct

  SELECT distinct Trip_distance, COUNT(Trip_distance) over(PARTITION by Trip_distance) as count_of_trips
  from sep_oct
  ORDER by 1

  ---   14. How many nulls are there in Ehail_fee? 

select sum(case when Ehail_fee is null then 1 else 0 end) as count_nulls,
       count(Ehail_fee) count_not_nulls 
from sep_oct

/*  15. Find the trips of which trip distance is greater than 15 miles (included) or less than 0.1 mile (included). 

It is possible to write this with only one where statement. However, this time write two queries and "union" them. 
The purpose of this question is to use union function. 
You can consider this question as finding outliers in a quick and dirty way, which you would do in your professional life too often. */

SELECT *
from sep_oct
WHERE Trip_distance >= 15
union 
SELECT *
from sep_oct
WHERE Trip_distance <= 0.1

--- or

select *
from sep_oct
where Trip_distance <= 0.1 or Trip_distance >= 15

/* 16. We would like to see the distribution (not like histogram) of Total_amount. 
Could you create buckets, or price range, for Total_amount and find how many trips there are in each buckets? 
Each range would be 5, until 35, i.e. 0-5, 5-10, 10-15 … 30-35, +35. The expected output would be as follows: */









