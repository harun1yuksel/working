-- SQL-6. ders_08.06.2022(session-5)

-- Ge�en dersten kalan yeri bitirece�iz
-- VIEWS
-- Database de SQL server�n y�netti�i db ler var. Bu bir disk alan�. Verilerimiz tablolarda bulunuyor
-- .. fiziksel olarak. Tablolarda ayn� �ekilde. Tablolar �zerine kurulan bir mimari var (dashborad vs) 
-- .. Bu tablolardan birinde bir de�i�iklik olursa �st katmanda b�y�k sorun olabilir.
-- .. Bu katman aras�nda stabil katmana ihtiyac�m�z var. Bu katman� sa�lad���m�z noktalardan birisi view
    --1.Bir tablonun g�r�nt�s�n� olu�turuyoruz. Bu g�r�nt� fiziksel olarak ayr� bir yer kaplam�yor. Tablolara
        -- .. ba�lant� sa�l�yor
    -- 2.Bir sorgumuz var. �ok uzun bu sorgu ama ihtiya� duyuyoruz diyelim. Bunu farkl� bir yerde bir kural olarak tan�mlarsak
       -- .. Bu view �zerinden daha rahat bir �ekilde sorgumuzu yapabiliriz
   -- CREATE VIEW view_name AS SELECT columns from tables [WHERE conditions];
-- Advantages of Views: Performance(bir uzun sorguyu view olarak kaydedip kullanma), Security, Storage, Simplicity

-- Soru: �r�n bilgilerini stok miktarlar� ile birlikte listeleyin demi�tik altta
SELECT	A.product_id, A.product_name, B.*
FROM	product.product A
LEFT JOIN product.stock B ON A.product_id = B.product_id
WHERE	A.product_id > 310;

-- Bu sorguyu bir view olarak kaydedelim
CREATE VIEW ProductStock AS
SELECT	A.product_id, A.product_name, B.*
FROM	product.product A
LEFT JOIN product.stock B ON A.product_id = B.product_id
WHERE	A.product_id > 310;

-- Hata. product_id 2 kere ge�iyor
CREATE VIEW ProductStock AS
SELECT	A.product_id, A.product_name, B.store_id,B.quantity
FROM	product.product A
LEFT JOIN product.stock B ON A.product_id = B.product_id
WHERE	A.product_id > 310;

-- Command completed sucsesfully
-- sampleretail-views-dbo.ProductStock

-- Bunu sorgular�m�n i�inde tablo olarak kullanabilirim
SELECT * FROM dbo.ProductStock
--Sorgu sonucunun ayn�s� geldi

-- Ko�ul da ekleyebiliriz
SELECT * FROM dbo.ProductStock
WHERE store_id=1

-- NOT: Bunu tek sorgu i�in yapabiliriz. Daha fazla sorgu i�in "procedure" kullanaca��z ilerde
-- NOT: ProductStock sadece bir script, as�l tabloyla olan bir ili�kisi var. Depolamada b�y�k katk�s� var
-- NOT: Bunu tablo olarak create edemez miyiz? Edebiliriz. O tablo fiziksel bir tablodur, dinamik bir tablo olmaz
-- NOT: Tablonun hep son durumu(de�i�meden �nceki(e�er de�i�tiyse)) ile ilgili bilgi almak istersem view kullanmal�y�z
-- NOT: VIEW i�erisinden ORDER BY kullanamay�z(VIEW OLU�MAYACAKTI). VIEW olu�tuktan sonra ORDER BY � kullanabiliriz
-- NOT: VIEW i�indeki sorgu i�in sampleretail-->view-->dbo.ProductStock--> sa� t�k-->design

---------------------------------
SELECT	A.product_id, A.product_name, B.store_id, B.quantity
INTO	#ProductStock
FROM	product.product A
LEFT JOIN product.stock B ON A.product_id = B.product_id
WHERE	A.product_id > 310;

SELECT * FROM #ProductStock;
-- Bu da diez ile bir ba�lant� ile olu�turulan ge�ici view. Ba�lant� kapan�nca bu gider


----------------------------

-- Hoca: Sizde buna VIEW yap�p al��t�rma yapal�m
-- Ma�aza �al��anlar�n� �al��t�klar� ma�aza bilgileriyle birlikte listeleyin
-- �al��an ad�, soyad�, ma�aza adlar�n� se�in
SELECT	A.first_name, A.last_name, B.store_name
FROM	sale.staff A
INNER JOIN sale.store B
	ON	A.store_id = B.store_id;
    
-- ��z�m
CREATE VIEW SaleStaff as
SELECT  A.first_name, A.last_name, B.store_name
FROM    sale.staff A
INNER JOIN sale.store B
    ON  A.store_id = B.store_id
    

--%% ADVANCED GROUPING FUNCTIONS
-- Table of Contents
-- Having clause
-- Grouping sets
-- Rollup
-- Cube
-- Pivot

--�rnek: bir ma�azada toplam ka� tane �r�n var(ma�aza baz�nda)
-- �rnek: A kategorisindeki �r�nlerin ortalama fiyat� vs
-- Bunlar� grouping functions la yap�yoruz
-- Group by + agregation kullanaca��z
/*
A   0     
B   5      
C   10
A   5            -----  A 15
B   10           ------ B 30
C   15           ------ C 45
A   10
B   15
C   20
*/


