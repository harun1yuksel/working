CREATE DATABASE Manufacturer
;

USE Manufacturer
;

CREATE SCHEMA Chocolate;

CREATE TABLE [Chocolate].[Product](
	[prod_id] [int] PRIMARY KEY NOT NULL,
	[prod_name] [nvarchar](50) NOT NULL,
	quantity INT NOT NULL,
	

	);

    CREATE TABLE [Chocolate].[Supplier](
        [supp_id] [int] PRIMARY KEY NOT NULL,
	[supp_name] [nvarchar](50) NOT NULL,
    	[supp_location] [nvarchar](50) NOT NULL,
	[supp_country] [nvarchar](50) NOT NULL,
    	[is_active] [bit] NOT NULL,

    )

CREATE TABLE [Chocolate].[Component](
    [comp_id] [int] PRIMARY KEY,
    [comp_name] [nvarchar](50) NOT NULL,
    [description] [nvarchar](50) NOT NULL,
    [quantity_comp] [int] NOT NULL, 

)

  CREATE TABLE [Chocolate].[Prod_Comp](
    [prod_id] [int] PRIMARY KEY NOT NULL, 
    [comp_id] [int] FOREIGN KEY REFERENCES [Chocolate].[Component] (comp_id),
    [quantity_comp] [int] NOT NULL, 
  )


CREATE TABLE [Chocolate].[Comp_Supp](
    [supp_id] [int] PRIMARY KEY NOT NULL,
    [comp_id] [int] NOT NULL FOREIGN KEY REFERENCES [Chocolate].[Component] (comp_id),
    [supp_country] [nvarchar](50) NOT NULL,
    [order_date] [date] NOT NULL, 
    [quantity] [int]
)

ALTER TABLE [Chocolate].[Prod_Comp] ADD CONSTRAINT FK2 FOREIGN KEY (comp_id) REFERENCES [Chocolate].[Component](comp_id);

ALTER TABLE [Chocolate].[Comp_Supp] ADD CONSTRAINT FK3 FOREIGN KEY (comp_id) REFERENCES [Chocolate].[Component](comp_id);
ALTER TABLE [Chocolate].[Supplier] ADD CONSTRAINT FK4 FOREIGN KEY (supp_id) REFERENCES [Chocolate].[Comp_Supp](supp_id);
