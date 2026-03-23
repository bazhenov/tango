- t3.micro

```
./criterion --save-baseline=main --confidence-level=0.99 --bench -n binary_search > /dev/null

while true; do
    ./criterion --bench --baseline=main --confidence-level=0.99 --measurement-time=1 -n -v binary_search >> ./criterion.txt

    ./criterion --save-baseline=paired --confidence-level=0.99 --measurement-time=1 --bench -n binary_search > /dev/null
    ./criterion --bench --baseline=paired --confidence-level=0.99 --measurement-time=1 -n -v binary_search >> ./criterion-paired.txt

    ./tango-1 compare ./tango-2 -t 1 -p -f binary_search >> ./tango.txt
done
```
