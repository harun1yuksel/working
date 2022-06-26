
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

SELECT 

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

SELECT * 
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
with A as (
    SELECT distinct MONTH(Order_Date) as order_month, cust_id, order_date,  
    row_number() over(PARTITION by cust_id order by MONTH(Order_Date)) as count_of_ordered_months
    From combined_table
   
)
select *, A.Cust_id as cust_id_ordeded_every_month
from A
Where count_of_ordered_months = 12



create VIEW cust_id_ordeded_every_month AS
    with A as (
        SELECT cust_id, order_date, MONTH(Order_Date) as order_month, 
        row_number() over(PARTITION by cust_id order by MONTH(Order_Date)) as count_of_ordered_months
        From combined_table
    
    )
    select A.Cust_id as cust_id_ordeded_every_month
    from A
    Where count_of_ordered_months = 12

---------------------

select count(*) custCount_ordered_jan_and_everyMonth 
from 
    (SELECT distinct cust_id as cust_ordered_jan, order_date, Customer_Name
    FROM combined_table
    where MONTH(Order_Date) = 1)   A,  cust_id_ordeded_every_month B 
    Where A.cust_ordered_jan = B.cust_id_ordeded_every_month


/* q6
Write a query to return for each user the time elapsed between the first
purchasing and the third purchasing, in ascending order by Customer ID
*/

select Cust_id, Customer_Name, Order_Date, 
lag(Order_Date, 2) over(PARTITION by cust_id order by Order_Date ) as third_order,
Datediff(day, lag(Order_Date, 2) over(PARTITION by cust_id order by Order_Date ), Order_Date) as days_between_3th_1th_order
from combined_table
order by cust_id

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
cast(round(1.0 * A.sum_11_14 / B.sum_products * 100, 2) as DECIMAL(3,1)) as per_sum11_14_to_sum_products
from A, B, cust_dimen C
where A.cust_id = B.cust_id and A.cust_id = C.cust_id


--------------------------------------------------------------------------
--------------------------------------------------------------------------

-------Customer Segmentation------

/* q1













