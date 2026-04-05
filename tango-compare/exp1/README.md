- t3.micro

```
while true; do
    ./criterion --bench --baseline=main -n -v >> ./criterion.txt

    ./criterion --save-baseline=paired --bench -n > /dev/null
    ./criterion --bench --baseline=paired -n -v >> ./criterion-paired.txt

    ./tango-1 compare ./tango-2 -t 1 -p >> ./tango.txt
    sleep 10
done
```
