from subprocess import call
import time
call(["make"])
# stride = 50
# max_step = 1000
# for i in range(0+stride,max_step+stride,stride):
#     call(["./app", str(i)])
#     print ("\n")

t = time.time()
tc = time.clock()
iterations = 10
for i in range(iterations):
    call(["./app", str(400)])
elapsed_time = time.time() - t
elapsed_clock = time.clock() -tc

average_time = elapsed_time / iterations
average_clock = elapsed_clock / iterations
print (average_time, average_clock)

