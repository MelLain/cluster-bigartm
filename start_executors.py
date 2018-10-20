import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vocab')
parser.add_argument('-b', '--batches-path')
parser.add_argument('-r', '--redis-addresses-path')
parser.add_argument('-n', '--num-executors')

parser.add_argument('-t', '--num-topics')
parser.add_argument('-i', '--num-inner-iter')
parser.add_argument('-c', '--continue-fitting')

parser.add_argument('-o', '--output-path')

def computeIndices(num_executors, size):
	step = int(size / num_executors)
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

	num_executors = int(args['num_executors']);
	token_indices = computeIndices(num_executors * len(redis_addresses), num_tokens)
	batch_indices = computeIndices(num_executors * len(redis_addresses), num_batches)

	cmd_str = './executor_main --num-topics {} --num-inner-iter {} --batches-dir-path {} --vocab-path {} --continue-fitting {} '.format(
    	args['num_topics'],
    	args['num_inner_iter'],
    	args['batches_path'],
    	args['vocab'],
    	args['continue_fitting'])

	executor_id = 0
	for addr in redis_addresses:
		additional_args = '--redis-ip {} --redis-port {} '.format(addr[0], addr[1])

		for _ in range(num_executors):
			additional_args += '--executor-id {} --token-begin-index {} --token-end-index {} --batch-begin-index {} --batch-end-index {}'.format(
				executor_id,
				token_indices[executor_id][0],
				token_indices[executor_id][1],
				batch_indices[executor_id][0],
				batch_indices[executor_id][1])

			executor_id += 1
		os.popen('{} {} &'.format(cmd_str, additional_args))

	with open(args['output_path'], 'w') as fout:
		for i in range(executor_id):
			fout.write('{}\n'.format(i))

if __name__ == '__main__':
	main()