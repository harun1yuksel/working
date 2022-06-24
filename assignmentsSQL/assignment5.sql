CREATE FUNCTION factorial(@number int)
RETURNS INT
as 

BEGIN
DECLARE @result INT = 1

while @number > 1
begin
    set @result = @number * @result
    set @number = @number - 1
    end


RETURN @result
END

select dbo.factorial(2)