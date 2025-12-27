-- Check the row counts for the most important tables
SELECT 'orders' AS table_name, COUNT(*) AS total_rows FROM orders
UNION ALL
SELECT 'items', COUNT(*) FROM order_items
UNION ALL
SELECT 'customers', COUNT(*) FROM customers
UNION ALL
SELECT 'products', COUNT(*) FROM products;


ALTER TABLE orders 
ALTER COLUMN order_purchase_timestamp TYPE TIMESTAMP 
USING order_purchase_timestamp::timestamp;

SELECT TO_CHAR(o.order_purchase_timestamp,'YYYY-MM') AS Time,
       ROUND(SUM(oi.price)::numeric,2) AS Total_Revenue_Per_Month
FROM orders AS o JOIN order_items AS oi ON o.order_id = oi.order_id
WHERE o.order_status = 'delivered'
GROUP BY o.order_purchase_timestamp
ORDER BY Time ASC;

-- Highest Month in profits most likely due to black friday
SELECT *
FROM(SELECT TO_CHAR(o.order_purchase_timestamp,'YYYY-MM') AS Time,
       ROUND(SUM(oi.price)::numeric,2) AS Total_Revenue_Per_Month
FROM orders AS o JOIN order_items AS oi ON o.order_id = oi.order_id
WHERE o.order_status = 'delivered'
GROUP BY o.order_purchase_timestamp
ORDER BY Time ASC) AS data
ORDER BY Total_Revenue_Per_Month DESC;




SELECT  pe.product_category_name_english AS Product_Category,
        ROUND(SUM(oi.price)::numeric,2) AS Total_Revenue
FROM order_items AS oi JOIN products AS p ON oi.product_id = p.product_id
                       JOIN product_category_name_translation AS pe ON p.product_category_name = pe.product_category_name
GROUP BY pe.product_category_name_english
ORDER BY Total_Revenue DESC
LIMIT 10;


SELECT c.customer_state AS State,
       ROUND(SUM(oi.price)::numeric,2) AS Total_Revenue,
	   ROUND(SUM(oi.freight_value)::numeric,2) AS Total_Freight
FROM customers AS c JOIN orders AS  o ON c.customer_id = o.customer_id
                    JOIN order_items  AS oi ON  o.order_id = oi.order_id
GROUP BY c.customer_state
ORDER BY  Total_Revenue DESC;

ALTER TABLE orders
ALTER COLUMN order_delivered_customer_date TYPE timestamp
USING order_delivered_customer_date:: timestamp;

SELECT c.customer_state AS State,
      ROUND(AVG(EXTRACT(DAY FROM(o.order_delivered_customer_date - o.order_purchase_timestamp)))::numeric,0) Days_to_Deliver
FROM customers AS c JOIN orders AS o ON c.customer_id = o.customer_id
WHERE o.order_status = 'delivered'
GROUP BY c.customer_state
ORDER BY Days_to_Deliver DESC;

SELECT  s.seller_id AS Seller_id,
        s.seller_state AS Seller_State,
		ROUND(SUM(oi.price):: numeric,2) AS Total_Revenue
FROM sellers AS s JOIN order_items AS oi ON s.seller_id = oi.seller_id
GROUP BY s.seller_id,s.seller_state
HAVING ROUND(SUM(oi.price):: numeric,2) > 100000
ORDER BY Total_Revenue DESC;



WITH Product_Revenue AS (
    SELECT 
        pe.product_category_name_english AS Category,
        p.product_id,
        SUM(oi.price) AS Total_Revenue
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    JOIN product_category_name_translation pe 
        ON p.product_category_name = pe.product_category_name
    GROUP BY Category,p.product_id
),
Ranked_Products AS (
    SELECT 
        Category,
        product_id,
        ROUND(Total_Revenue:: numeric,2),
        DENSE_RANK() OVER(PARTITION BY Category ORDER BY Total_Revenue DESC) AS Rank_In_Category
    FROM Product_Revenue
)
SELECT * FROM Ranked_Products
WHERE Rank_In_Category <= 3
ORDER BY Category, Rank_In_Category;





CREATE OR REPLACE VIEW analytics_master AS
SELECT 
    o.order_id,
    o.order_purchase_timestamp,
    oi.price,
    oi.freight_value,
    p.product_category_name,
    c.customer_city,
    c.customer_state,
    s.seller_city,
    s.seller_state
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
JOIN customers c ON o.customer_id = c.customer_id
JOIN sellers s ON oi.seller_id = s.seller_id
WHERE o.order_status = 'delivered';