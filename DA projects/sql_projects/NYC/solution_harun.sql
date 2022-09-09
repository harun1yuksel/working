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
Each range would be 5, until 35, i.e. 0-5, 5-10, 10-15 … 30-35, +35. The expected output would be as follows: 
   
   Payment_range            Trip_count
     0-5                        47
     10-15                      938
     15-20                      214   etc.

     */
         
SELECT *
from sep_oct

select *,
case when Total_amount BETWEEN 0 and 5 then '0-5'
     when Total_amount > 5  and Total_amount <= 10 then '6-10'
    when Total_amount > 10  and Total_amount <= 15 then '11-15'
    when Total_amount > 15  and Total_amount <= 20 then '16-20'
    when Total_amount > 20  and Total_amount <= 25 then '21-25'
    when Total_amount > 25  and Total_amount <= 30 then '26-30'
    when Total_amount > 30  and Total_amount <= 35 then '31-35'
    when Total_amount > 35 then '35+'
    When Total_amount < 0 then 'negative_payment'
    end as Payment_range
from sep_oct


with A as (
    select
case when Total_amount >= 0 and Total_amount <= 5 then '0-5'
     when Total_amount > 5  and Total_amount <= 10 then '6-10'
    when Total_amount > 10  and Total_amount <= 15 then '11-15'
    when Total_amount > 15  and Total_amount <= 20 then '16-20'
    when Total_amount > 20  and Total_amount <= 25 then '21-25'
    when Total_amount > 25  and Total_amount <= 30 then '26-30'
    when Total_amount > 30  and Total_amount <= 35 then '31-35'
    when Total_amount > 35 then '35+'
    When Total_amount < 0 then 'negative_payment'
    end as Payment_range
from sep_oct
)
select payment_range, COUNT(Payment_range) as trip_count
from A 
GROUP by (Payment_range)
order by 1;

/*     17. We also would like to analyze the performance of each driver’s earning. 
Could you add driver_id to payment distribution table?  */

with A as (
    select *, 
case when Total_amount >= 0 and Total_amount <= 5 then '0-5'
     when Total_amount > 5  and Total_amount <= 10 then '6-10'
     when Total_amount > 10  and Total_amount <= 15 then '11-15'
     when Total_amount > 15  and Total_amount <= 20 then '16-20'
     when Total_amount > 20  and Total_amount <= 25 then '21-25'
     when Total_amount > 25  and Total_amount <= 30 then '26-30'
     when Total_amount > 30  and Total_amount <= 35 then '31-35'
     when Total_amount > 35 then '35+'
     When Total_amount < 0 then 'negative_payment'
     end as Payment_range
from sep_oct
)
select  distinct driver_id, payment_range, 
COUNT(Payment_range) over(PARTITION by driver_id, Payment_range) as driver_trip_count,
Cast(sum(Total_amount) over(PARTITION by driver_id, payment_range) as Decimal (10,2)) as driver_earning_for_range,
Cast((sum(Total_amount) over(PARTITION by driver_id)) as Decimal (10, 2)) as driver_earning_total
from A 
order by 1;

/*    18. Could you find the highest 3 Total_amount trips for each driver? Hint: Use “Window” functions. */

WITH A as (
SELECT driver_id, Total_amount,
ROW_NUMBER() OVER(partition by driver_id order by Total_amount desc) as row_num
from sep_oct
)
SELECT *
from A
WHERE row_num <=3;

WITH A as (
SELECT *,
ROW_NUMBER() OVER(partition by driver_id order by Total_amount desc) as row_num
from sep_oct
)
SELECT *
from A
WHERE row_num <=3;

/*    19. Could you find the lowest 3 Total_amount trips for each driver? Hint: Use “Window” functions. */

WITH A as (
SELECT *,
ROW_NUMBER() OVER(partition by driver_id order by Total_amount Asc) as row_num
from sep_oct
)
SELECT *
from A
WHERE row_num <=3;

/*    20. Could you find the lowest 10 Total_amount trips for driver_id 1? 
Do you see any anomaly in the rank? (same rank, missing rank etc). 
Could you “fix” that so that ranking would be 1, 2, 3, 4… (without any missing rank)? 
Note that 1 is the lowest Total_amount in this question. 
Also, same ranks would continue to exist since there might be the same Total_amount. Hint: dense_rank. */

WITH A as (
SELECT *,
ROW_NUMBER() OVER(partition by driver_id order by Total_amount Asc) as row_num
from sep_oct
)
SELECT *
from A
WHERE row_num <=10 and driver_id = '1'

/*    21. Our friend, driver_id 1, is very happy to see what we have done for her 
(Yes, it is “her”. Her name is Gertrude Jeannette, https://en.wikipedia.org/wiki/Gertrude_Jeannette. That is why her id is 1). 
Could you do her a favor and track her earning after each trip? */

SELECT Total_amount,
sum(Total_amount) OVER(order by lpep_pickup_datetime, unbounded preceding) as cumulative_sum
from sep_oct
WHERE driver_id = '1'

















