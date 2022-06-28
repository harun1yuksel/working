SELECT product_name, product_id
FROM product.product
WHERE product_name = '2TB Red 5400 rpm SATA III 3.5 Internal NAS HDD'


SELECT product_id, order_id
FROM [sale].[order_item]
WHERE product_id = 6

SELECT A.customer_id, B.product_id
FROM	sale.orders A
INNER JOIN	sale.order_item B
	ON	A.order_id = B.order_id
WHERE B.product_id = 6

CREATE VIEW customers_pr_id6 AS
SELECT A.customer_id, B.product_id
FROM	sale.orders A
INNER JOIN	sale.order_item B
	ON	A.order_id = B.order_id
WHERE B.product_id = 6

SELECT *
From customers_pr_id6

SELECT product_name, product_id
FROM product.product
WHERE product_name in ('Polk Audio - 50 W Woofer - Black', 
                        'SB-2000 12 500W Subwoofer (Piano Gloss Black)',
                         'Virtually Invisible 891 In-Wall Speakers (Pair)')



SELECT order_id, product_id
FROM sale.order_item
WHERE sale.order_item.product_id in (13, 16, 21)

CREATE VIEW customer_13_16_21 AS
SELECT A.customer_id, B.product_id
FROM	sale.orders A
INNER JOIN	sale.order_item B
	ON	A.order_id = B.order_id
WHERE B.product_id in (13, 16, 21)

select *
from customer_13_16_21

SELECT A.customer_id, A.product_id, B.product_id
FROM customer_13_16_21 A
INNER JOIN cutomers_pr_id6 B 
    on A.customer_id = B.customer_id

SELECT A.product_id, COUNT(*) AS num_sale_with_product6
FROM customer_13_16_21 A
INNER JOIN cutomers_pr_id6 B 
    on A.customer_id = B.customer_id
GROUP BY A.product_id 

