import tensorflow as tf
import warnings


a = tf.constant([[1, 2, 3],
                 [4, 5, 6.4],
                 ])
print()

print(a**3)
type(a)
print(a.shape)
print(a.dtype)

@tf.function
def my_func(x):
  print('Tracing.\n')
  return tf.reduce_sum(x)

print("Result: {}".format(my_func([2,3,4])))


if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")
