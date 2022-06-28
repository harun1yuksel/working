SELECT * 
from sale.order_item

select *, sum(quantity) OVER(partition by product_id, discount)
from sale.order_item


SELECT distinct product_id, discount, 
sum(quantity) OVER(partition by product_id, discount) amount_of_sale_acordingto_discount
from sale.order_item


