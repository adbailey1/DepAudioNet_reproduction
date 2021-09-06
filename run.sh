#! /usr/bin/env bash

FILES_PATH='/path/to/depaudionet/code'
PYTHON_ENV='/path/to/Python_Env'

#WITH GPU
count=1
end=1
while [ $count -le $end ]
do
    PTH=${FILES_PATH}/"main"${count}".py"
    echo "$PTH"
    echo "$PYTHON_ENV"
    $PYTHON_ENV "$PTH" train --validate --vis --debug --cuda --position=${count}
    count=$(($count + 1))
done
 #python3.7 $FILES_PATH/main1.py train --validate --vis --cuda --debug
