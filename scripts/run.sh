while true; do                                                                                                                                                                                   
python gen_eval.py generate --input data/testset/selected_800.json --output test_trajectory_for_benchmark --model gpt-5-mini                                                                                                                                   
if [ $? -eq 0 ]; then                                                                                                                                                                          
    echo "Done!"                                                                                                                                                                                 
    break                                                                                                                                                                                        
fi                                                      
echo "Script crashed. Sleeping 5 minutes then retrying..."
sleep 300                                                                                                                                                                                      
done