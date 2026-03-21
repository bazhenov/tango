- t3.micro

```
./criterion --save-baseline=main --bench -n > /dev/null

while true; do
    ./criterion --bench --baseline=main --measurement-time=1 -n -v >> ./criterion.txt

    ./criterion --save-baseline=in-place --measurement-time=1 --bench -n > /dev/null
    ./criterion --bench --baseline=in-place --measurement-time=1 -n -v >> ./criterion-in-place.txt

    ./tango-1 compare ./tango-2 -t 1 >> ./tango.txt
done
```
