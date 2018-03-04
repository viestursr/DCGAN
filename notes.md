### Get MNIST set
```
mnist = input_data.read_data_sets("MNIST_data/")
```

### Get MNIST batch
```
image_batch = mnist.train.next_batch(batch_size)
```

### Reshape one flat image to 28x28
```
sample_image = image.reshape([28, 28])
```

### Show image
```
plt.imshow(sample_image, cmap='Greys')
plt.show()
```

### Save image
```
plt.imsave(fname=output_path + str(batch_number) + ".png", arr=sample_image, cmap='Greys')
```