
    UPDATE orders_dimen
    set order_date = concat(SUBSTRING(order_date,7,4),'-' , SUBSTRING(order_date,4,2),'-', SUBSTRING(order_date, 1,2))
    where order_date like '[0-9][0-9]-[0-9][0-9]-[0-9][0-9][0-9][0-9]'

    alter table orders_dimen
    ALTER COLUMN order_date date;

    UPDATE shipping_dimen
    set Ship_Date = concat(SUBSTRING(Ship_Date,7,4),'-' , SUBSTRING(Ship_Date,4,2),'-', SUBSTRING(Ship_Date, 1,2))
    where Ship_Date like '[0-9][0-9]-[0-9][0-9]-[0-9][0-9][0-9][0-9]'

    alter table shipping_dimen
    ALTER COLUMN Ship_Date date; 

   
SELECT *
from market_fact
ORDER by ord_id


update market_fact
set Cust_id = replace(Cust_id, 'Cust_', '')

UPDATE market_fact
set Ord_id = replace(Ord_id, 'Ord_', '')

update market_fact
set Prod_id = replace(Prod_id, 'Prod_', '')

SELECT *
from cust_dimen

update cust_dimen
set Cust_id = replace(Cust_id, 'Cust_', '')



update market_fact
set Ship_id = replace(Ship_id, 'SHP_', '')



SELECT *
from orders_dimen

UPDATE orders_dimen
set Ord_id = replace(Ord_id, 'Ord_', '')

SELECT *
from prod_dimen

update prod_dimen
set Prod_id = replace(Prod_id, 'Prod_', '')

select * 
from shipping_dimen

update shipping_dimen
set Ship_id = replace(Ship_id, 'SHP_', '')

------Analyze the data by finding the answers to the questions below-----
/* 
q1
Using the columns of “market_fact”, “cust_dimen”, “orders_dimen”,
“prod_dimen”, “shipping_dimen”, Create a new table, named as
“combined_table”. */

SELECT MF.Ord_id, MF.[Prod_id],MF.[Ship_id], MF.[Cust_id], MF.[Sales], MF.[Discount], MF.[Order_Quantity], MF.[Product_Base_Margin],
    CD.Customer_Name, CD.Province, CD.Region, CD.[Customer_Segment],
    OD.[Order_Date], OD.[Order_Priority],
    PD.[Product_Category], PD.[Product_Sub_Category],
    SD.[Order_ID], SD.[Ship_Mode], SD.[Ship_Date]
    into combined_table
    from market_fact MF
    LEFT JOIN cust_dimen CD on MF.Cust_id = CD.Cust_id
    LEFT JOIN orders_dimen OD on MF.ord_id = OD.ord_id
    left join prod_dimen PD on MF.Prod_id = PD.Prod_id
    left join shipping_dimen SD on MF.Ship_id = SD.Ship_id

/* q2 
Find the top 3 customers who have the maximum count of orders.
*/

SELECT cust_id, sum(Order_Quantity) sum_order_per_cus
from combined_table
GROUP BY Cust_id
ORDER By sum_order_per_cus DESC


WITH A AS
    (SELECT Top 3 Cust_id, sum(Order_Quantity) sum_order_per_cus
    from combined_table
    GROUP BY Cust_id
    ORDER By sum_order_per_cus DESC)

SELECT A.cust_id, B.Customer_Name, A.sum_order_per_cus
from A, cust_dimen B
WHERE A.cust_id = B.cust_id

/* q3
Create a new column at combined_table as DaysTakenForDelivery that
contains the date difference of Order_Date and Ship_Date.
*/

SELECT Datediff(day, Order_Date, Ship_Date)
from combined_table

alter table combined_table add DaysTakenForDelivery as Datediff(day, Order_Date, Ship_Date)

SELECT Cust_id, Order_ID, DaysTakenForDelivery
from combined_table

/*q4
Find the customer whose order took the maximum time to get delivered.
*/

    SELECT Cust_id, Customer_Name, DaysTakenForDelivery
    from combined_table
    ORDER by DaysTakenForDelivery DESC



    SELECT top 1 Cust_id, Customer_Name, DaysTakenForDelivery
    from combined_table
    ORDER by DaysTakenForDelivery DESC

/*q5 
Count the total number of unique customers in January and how many of them
came back every month over the entire year in 2011
*/

SELECT distinct cust_id as cust_ordered_jan, order_date
FROM combined_table
where MONTH(Order_Date) = 1 

------
SELECT distinct Cust_id, MONTH(order_date) as order_month
from combined_table

with A as (
    SELECT distinct Cust_id, MONTH(order_date) as order_month
from combined_table
)

SELECT Cust_id, order_month,
row_number() over(PARTITION by cust_id order by order_month) num_of_ordered_month
from A

/* yukarıdaki tabloya göre her ay sipariş veren müşteri bulunamadı, yada ben bulamadım :) */ 



/* q6
Write a query to return for each user the time elapsed between the first
purchasing and the third purchasing, in ascending order by Customer ID
*/
with A as (
select distinct Cust_id, Customer_Name, Order_Date, 
DENSE_RANK() over(PARTITION by cust_id order by Order_Date ) as denseRank_num
from combined_table
)
select *,
lag(order_date) over(PARTITION by cust_id order by Order_Date ) as previous_order,
Datediff(day, lag(order_date) over(PARTITION by cust_id order by Order_Date), order_date)  as days_between_orders
from A

SELECT cust_id, order_date
from combined_table


CREATE VIEW days_between_orders AS
        with A as (
            select distinct Cust_id, Customer_Name, Order_Date, 
            DENSE_RANK() over(PARTITION by cust_id order by Order_Date ) as denseRank_num   ---bu tablo distinct verileri taşıyor fonksiyonları buraya yazınca distinct olmuyor.
                                                                        ----- çünkü yeni oluşan sütünlar distinktliği bozuyor ve aynı order date li satırlar geliyor.

            from combined_table
        )
        SELECT *, lag(Order_Date, 2) over(PARTITION by cust_id order by Order_Date ) as two_previous_order,
                Datediff(day, lag(Order_Date, 2) over(PARTITION by cust_id order by Order_Date ), Order_Date) as days_between_3th_and_1st_ord
        from A

SELECT *
from days_between_orders
where denseRank_num = 3

    ROW_NUMBER() over(PARTITION by cust_id order by Order_Date ) as row_num

    Datediff(day, lag(Order_Date, 3) over(PARTITION by cust_id order by Order_Date ), Order_Date) as days_between_3th_and_1st_ord,




/* q7

Write a query that returns customers who purchased both product 11 and
product 14, as well as the ratio of these products to the total number of
products purchased by the customer.

*/

select *
from combined_table

with A AS (
        SELECT Cust_id, order_quantity as quantity11, Customer_Name
        from combined_table
        where Prod_id = 11
),
      B AS (

        SELECT Cust_id, order_quantity as quantity14, Customer_Name
        from combined_table
        where Prod_id = 14
    
)

Select A.Cust_id, A.Customer_Name, A.quantity11, B.quantity14, A.quantity11 + B.quantity14 as quantity_11_14
FROM A, B 
WHERE A.Cust_id = B.Cust_id


create VIEW cus_11_14 AS

       with A AS (
        SELECT Cust_id, order_quantity as quantity11, Customer_Name
        from combined_table
        where Prod_id = 11
),
      B AS (

        SELECT Cust_id, order_quantity as quantity14, Customer_Name
        from combined_table
        where Prod_id = 14
    
)

    Select A.Cust_id, A.Customer_Name, A.quantity11, B.quantity14, A.quantity11 + B.quantity14 as quantity_11_14
    FROM A, B 
    WHERE A.Cust_id = B.Cust_id


SELECT * 
from cus_11_14


WITH A AS (

    SELECT cust_id, sum(quantity_11_14) sum_11_14
    from cus_11_14 
    Group by cust_id
),
B AS (
    SELECT cust_id, sum(Order_Quantity) sum_products
    from combined_table
    Group by cust_id
)

SELECT A.cust_id, C.Customer_Name, A.sum_11_14, B.sum_products, 
cast(1.0 * A.sum_11_14 / B.sum_products * 100 as numeric (3,1)) as per_sum11_14_to_sum_products
from A, B, cust_dimen C
where A.cust_id = B.cust_id and A.cust_id = C.cust_id


--------------------------------------------------------------------------
--------------------------------------------------------------------------

-------Customer Segmentation------

/* 
Categorize customers based on their frequency of visits.
*/

SELECT distinct cust_id, order_date
from combined_table   --- devam eden hesaplarda bu tablo kullanılacak böylece aynı gün verilen siparişler tek sipariş gibi heaplama yapılcak.

with A as (
    SELECT distinct cust_id, order_date
    from combined_table 
    )

SELECT cust_id, Order_Date, 
lag(Order_Date) over(PARTITION by cust_id order by order_date) as previous_visit,
count(*) over(PARTITION by cust_id) as num_of_visits
from A;


with A as (
    SELECT distinct cust_id, order_date
    from combined_table 
    )


SELECT cust_id, Order_Date, 
lag(Order_Date) over(PARTITION by cust_id order by order_date) as previous_visit ,
DATEDIFF(DAY, lag(Order_Date) over(PARTITION by cust_id order by order_date), Order_Date) as days_between_visits,
count(*) over(PARTITION by cust_id) as num_of_visits
from A;

create view diff_of_visits AS 
    with A as (
        SELECT distinct cust_id, order_date
        from combined_table 
        )


    SELECT cust_id, Order_Date, 
    lag(Order_Date) over(PARTITION by cust_id order by order_date) as previous_visit ,
    DATEDIFF(DAY, lag(Order_Date) over(PARTITION by cust_id order by order_date), Order_Date) as days_between_visits,
    count(*) over(PARTITION by cust_id) as num_of_visits
    from A;


CREATE VIEW visit_frequency As
    SELECT *,
    avg(days_between_visits) over(PARTITION by cust_id) as avg_visit_frequency_days
    from diff_of_visits;

SELECT *
from visit_frequency


---- 'regular and frequent' : ziyare sayısı 3 ten fazla ve ve iki ziyaret arasındaki ortlama gün sayısı 60 ve altında ise
---- 'regular but not frequent' : ziyare sayısı 3 ten fazla ama iki ziyaret arasındaki ortlama gün sayısı 60 ın üzerinde ise
---- 'not regular' : ziyare sayısı 3 ten az ise

SELECT *,
case 

    when num_of_visits >= 3 and avg_visit_frequency_days <= 60 then 'regular and frequent' 
    when num_of_visits >= 3 and avg_visit_frequency_days > 60 then 'regular but not frequent'
    else 'not regular' END as frequency_of_visits

from visit_frequency

-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------

/* Month-Wise Retention Rate */



SELECT cust_id, order_date,
   DATEADD(MONTH, DATEDIFF(MONTH, '2000-01-01', order_date), convert(date, '2000-01-01')) AS year_month --- bu satır siparişlerin sadece yılını ve ayını gösteren bir sütün oluşturuyor.
FROM combined_table; --- bir ay içinde ki herhangi bir gün için aynı sonucu döndürüyor böylece o yıl ve ayı döndürüyor gün sabit 01 kalıyor. 


alter table combined_table 
add year_month 
AS DATEADD(MONTH, DATEDIFF(MONTH, '2000-01-01', order_date), convert(date, '2000-01-01'))




SELECT cust_id, order_date, year_month
FROM combined_table ---aynı ay içinde aynı müşteriye ait kayıtlar var mesela 4 . müşteri


SELECT count(cust_id), year_month
FROM combined_table
Group by year_month ------aynı ay içinde aynı müşteriye ait kayıtlar var. bu yüzden bu tablodaki toplam rakamları aşağıdakinden büyük çıkıyor.


SELECT count(distinct cust_id) count_of_customers_monthly, year_month
FROM combined_table
Group by year_month; -----distinct ile saydırınca her ay toplam kaç farklı müşteri olduğunu verdi.


with A as (
    SELECT count(distinct cust_id) count_of_customers_monthly, year_month
FROM combined_table
Group by year_month
)

SELECT distinct B.Cust_id, B.Order_Date, count_of_customers_monthly, A.year_month
From A, combined_table B
WHERE A.year_month = B.year_month;

CREATE VIEW monthly_cust_count AS
    with A as (
    SELECT count(distinct cust_id) count_of_customers_monthly, year_month
    FROM combined_table
    Group by year_month
    )

    SELECT distinct B.Cust_id, B.Order_Date, count_of_customers_monthly, A.year_month
    From A, combined_table B
    WHERE A.year_month = B.year_month;


SELECT Cust_id, order_date, 
        lag(order_date) over(PARTITION by cust_id order by order_date) as previous_order_date, 
        year_month, count_of_customers_monthly
from monthly_cust_count;


SELECT Cust_id, order_date, 
        lag(order_date) over(PARTITION by cust_id order by order_date) as previous_order_date, 
        DATEDIFF(MONTH, lag(order_date) over(PARTITION by cust_id order by order_date), order_date) as months_between_orders,
        year_month, count_of_customers_monthly
from monthly_cust_count;


CREATE VIEW cust_in_2consecutive_months AS
    with A AS(
        SELECT Cust_id, order_date, lag(order_date) over(PARTITION by cust_id order by order_date) as previous_order_date, 
            DATEDIFF(MONTH, lag(order_date) over(PARTITION by cust_id order by order_date), order_date) as months_between_orders,
            year_month, count_of_customers_monthly
        from monthly_cust_count
    )

        SELECT *
        from A
        where months_between_orders = 1;

SELECT *
from cust_in_2consecutive_months --- bu tablo yukarıda ki where şartı ile peşpeşe iki ay alışveriş yapan müşterileri getiriyor. 
--- bu müşterilerin aylık toplam sayısını o ayın retained müşterileri olarak tanımlayacağız. (aşağıdaki tabloda ki 'monthly_count_of_retained_cust' sütünu)




SELECT *, count(months_between_orders) over(PARTITION by year_month) as monthly_count_of_retained_cust
from cust_in_2consecutive_months;


with A As (
    SELECT *, count(months_between_orders) over(PARTITION by year_month) as monthly_count_of_retained_cust
    from cust_in_2consecutive_months
)

select distinct year_month, monthly_count_of_retained_cust, count_of_customers_monthly,
cast(round(1.0 *  monthly_count_of_retained_cust / count_of_customers_monthly * 100, 2) as decimal(3,  1)) as monthwise_cust_retention_rate
from A;


----- kısa çözüm 

create view tbl_by_time as
select distinct cust_id, year(order_date) years, month(order_date) months
	, dense_rank() over(order by year(order_date),month(order_date) ) rank_by_time
from combined_table

drop table if exists result_table
create table result_table (
	years int,
	months int,
	monthly_rate decimal(10,2)
)
----------
---------
declare	 @rank_min int
		,@rank_max int
		,@result decimal(10,2)

select @rank_min = min(rank_by_time) from tbl_by_time
select @rank_max = max(rank_by_time) from tbl_by_time

while @rank_min < @rank_max
begin
	with t1 as(
	select cust_id
	from tbl_by_time
	where	rank_by_time = @rank_min
	intersect 
	select cust_id
	from tbl_by_time
	where	rank_by_time = @rank_min+1
	) 
	select @result = (1.0*count(*)/(select count(*) from tbl_by_time where rank_by_time = @rank_min+1))
	from t1	
insert into result_table 
values ( (select distinct years from tbl_by_time where rank_by_time=@rank_min+1)
		,(select distinct months from tbl_by_time where rank_by_time=@rank_min+1)
		,@result
		)
set @rank_min += 1
end


-------fonksiyon bitti ve tabloyu yazdırdık
select * from result_table


