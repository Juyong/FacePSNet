# Before running you can adjust contents in '[]' according to your actual situation

python render_depth.py 0 list.txt
# python render_depth.py [gpu_id] [list_path]

./process_normal list.txt
# ./process_normal [list_path]

python opt_depth.py 0 0.1 30000 list.txt
# python opt_depth.py [gpu_id] [learning_rate] [iteration_time] [list_path]

./view_depth list.txt
# ./view_depth [list_path]

# After running the 'depth.obj' will be generated and stored under the input path
