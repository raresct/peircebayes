cores=$(fgrep -c processor /proc/cpuinfo)
echo $cores
# executes pb in parallel
xargs --arg-file=test_parallel.sh \
      --max-procs=$cores  \
      --replace \
      --verbose \
      /bin/sh -c "{}"
# merge plates

