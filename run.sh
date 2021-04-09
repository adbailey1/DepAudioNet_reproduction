#! /usr/bin/env bash

FILES_PATH='/path/to/dir/depaudionet'
PYTHON_ENV='/path/to/python/env'

#WITH GPU
count=1
end=2
while [ $count -le $end ]
do
    PTH=${FILES_PATH}/"main"${count}".py"
    echo "$PTH"
    echo "$PYTHON_ENV"
    $PYTHON_ENV "$PTH" train --validate --vis --cuda --debug --position=${count}
    #python3.7 "$PTH" train --validate --vis --cuda --debug --position=${count}
    count=$(($count + 1))
done
 #python3.7 $FILES_PATH/main1.py train --validate --vis --cuda --debug
