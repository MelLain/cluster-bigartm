import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vocab')
parser.add_argument('-b', '--batches-path')
parser.add_argument('-r', '--redis-addresses-path')
parser.add_argument('-n', '--num-executor-threads')

parser.add_argument('-t', '--num-topics')
parser.add_argument('-i', '--num-inner-iter')
parser.add_argument('-c', '--continue-fitting')
parser.add_argument('-p', '--cache-phi')

def ceil(number):
    z = int(number)
    if z < number:
        z += 1
    return z

def computeIndices(num_executors, size):
    step = ceil(size / float(num_executors))
    result = []
    for i in range(num_executors):
        result.append((step * i, min(step * (i + 1), size)))
    return result

def main():
	args = vars(parser.parse_args())

	redis_addresses = []
	for line in open(args['redis_addresses_path']):
		redis_addresses.append(line.strip().split(' '))
	print 'Number of instances is {}'.format(len(redis_addresses))

	num_tokens = int(os.popen('wc -l {}'.format(args['vocab'])).read().strip().split(' ')[0])
	num_batches = int(os.popen('ls -lt {} | wc -l'.format(args['batches_path'])).read().strip().split(' ')[0]) - 1
	print 'Number of tokens: {}'.format(num_tokens)
	print 'Number of batches: {}'.format(num_batches)

	token_indices = computeIndices(len(redis_addresses), num_tokens)
	batch_indices = computeIndices(len(redis_addresses), num_batches)

	assert token_indices[0][0] == 0
	assert token_indices[-1][-1] == num_tokens
	assert batch_indices[0][0] == 0
	assert batch_indices[-1][-1] == num_batches

	cmd_str = './executor_main --num-topics {} --num-inner-iter {} --batches-dir-path {} --vocab-path {} --continue-fitting {} --cache-phi {}'.format(
    	args['num_topics'],
    	args['num_inner_iter'],
    	args['batches_path'],
    	args['vocab'],
    	args['continue_fitting'],
    	args['cache_phi'])

	for executor_id, addr in enumerate(redis_addresses):
		additional_args = '--redis-ip {} --redis-port {} --num-threads {} '.format(addr[0], addr[1], int(args['num_executor_threads']))
		additional_args += '--executor-id {} --token-begin-index {} --token-end-index {} --batch-begin-index {} --batch-end-index {}'.format(
			executor_id,
			token_indices[executor_id][0],
			token_indices[executor_id][1],
			batch_indices[executor_id][0],
			batch_indices[executor_id][1])

		print '{} {} &'.format(cmd_str, additional_args)
		os.popen('{} {} &'.format(cmd_str, additional_args))

if __name__ == '__main__':
	main()
