----Use pickup information instead of drop-off, when necessary:
/*
   1. Most expensive trip (total amount).
   2. Most expensive trip per mile (total amount/mile).
   3. Most generous trip (highest tip).
   4. Longest trip duration.
   5. Mean tip by hour.
   6. Median trip cost (This question is optional. You can search for “median” calculation if you want).
   7. Average total trip by day of week (Fortunately, we have day of week information. Otherwise, we need to create a new date column without hours from date column. Then, we need to create "day of week" column, i.e Monday, Tuesday .. or 1, 2 ..,  from that new date column. Total trip count should be found for each day, lastly average total trip should be calculated for each day).
   8. Count of trips by hour (Luckily, we have hour column. Otherwise, a new hour column should be created from date column, then count trips by hour).
   9. Average passenger count per trip.
  10. Average passenger count per trip by hour.
  11. Which airport welcomes more passengers: JFK or EWR? Tip: check RateCodeID from data dictionary for the definition (2: JFK, 3: Newark).
  12. How many nulls are there in Total_amount?
  13. How many values are there in Trip_distance? (count of non-missing values)
  14. How many nulls are there in Ehail_fee? 
  15. Find the trips of which trip distance is greater than 15 miles (included) or less than 0.1 mile (included). It is possible to write this with only one where statement. However, this time write two queries and "union" them. The purpose of this question is to use union function. You can consider this question as finding outliers in a quick and dirty way, which you would do in your professional life too often.
  16. We would like to see the distribution (not like histogram) of Total_amount. Could you create buckets, or price range, for Total_amount and find how many trips there are in each buckets? Each range would be 5, until 35, i.e. 0-5, 5-10, 10-15 … 30-35, +35. The expected output would be as follows:



    17. We also would like to analyze the performance of each driver’s earning. Could you add driver_id to payment distribution table?  The expected output would be as follows:



Note that there are more rows in this table. 

   18. Could you find the highest 3 Total_amount trips for each driver? Hint: Use “Window” functions.
   19. Could you find the lowest 3 Total_amount trips for each driver? Hint: Use “Window” functions.
   20. Could you find the lowest 10 Total_amount trips for driver_id 1? Do you see any anomaly in the rank? (same rank, missing rank etc). Could you “fix” that so that ranking would be 1, 2, 3, 4… (without any missing rank)? Note that 1 is the lowest Total_amount in this question. Also, same ranks would continue to exist since there might be the same Total_amount. Hint: dense_rank.
   21. Our friend, driver_id 1, is very happy to see what we have done for her (Yes, it is “her”. Her name is Gertrude Jeannette, https://en.wikipedia.org/wiki/Gertrude_Jeannette. That is why her id is 1). Could you do her a favor and track her earning after each trip? She would be very thankful if we can provide her with the following information:


 Hint: Cumulative sum, running total

   22. Gertrude is fascinated by your work and would like you to find max and min Total_amount. She is ok with the following:


   23. There is one thing that Gertrude could not understand. Min Total_amount is 0, however we did not show any 0 while we track her earning (in cumulative sum question). It seems we owe her an explanation. Why do you think this happened?

Use October data for Q24 - Q31:
   24. Is there any new driver in October? Hint: Drivers existing in one table but not in another table.
   25. Total amount difference between October and September.
   26. Revenue of drivers each month.






   27. Trip count of drivers each month.



   28. Revenue_per-trip of drivers each month.



   29. Revenue per day of week comparison.












   30. Revenue per day of week for each driver comparison



  31. Revenue and trip count comparison of VendorID. You can also add passenger count, trip mile etc as a practice for yourself.



Use September data for Q32 - Q35:
 32 Find the trips that are longer than previous trip. Tip: Luckily, trips are sorted by date and trip IDs are consistent with date. So, we are ready to do "SELF JOIN". You should use trip id to join table with itself. The trick is each trip should be joined with the previous trip. That means join key would be table1.trip_id = table2.trip_id - 1 (or table1.trip_id = table2.trip_id + 1. This is another trick, think about it :) ). Then, compare the duration of two trips.  
  33. For driver ID 1, find the trips that are shorter than the successor (next) trip? 
  34. Which drivers are having good days? :)  (These are the drivers whose next trip is longer than previous trip. In other words, trip duration would increase by every trip for the driver). 
  35. Could you solve Q34 for total amount instead of trip duration.



  Information about the data columns (e.g., the data dictionary) are provided in: https://data.cityofnewyork.us/api/views/hvrh-b6nb/files/65544d38-ab44-4187-a789-5701b114a754?download=true&filename=data_dictionary_trip_records_green.pdf
  
  
Note: All these questions are very applicable in business life. Questions after Q32 would look harder (Yes, it is!), however they are essential in some business domains. For example, Netflix and Amazon Prime Video measure customer engagement. The way to find out more engaged customers is comparing each session duration, i.e. any time you log in, with previous/next session. This business problem is a version of Q33.
How about measuring popularity of a movie? Calculate total minutes watched for each movie each day. Compare total minutes watched previous/next day for each movie. If a movie is attracting less viewer, i.e less minutes watched each day, then it is time to end the contract for this movie. This business question is a version of Q31, Q34 and Q35. 
*/

