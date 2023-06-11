

for (( i=1;i<=50;i++ )) do
       echo "start to run of $i"
       # sleep 300
       nohup python workflow.py &> log.txt &
       echo "start to sleep run of $i"
       sleep 300
       echo "sleep done. end run of $i"
       ps -ef | egrep "workflow.py" | grep -v grep |  awk '{ print $2 }' | xargs kill -9
       echo "kill $i"
done



