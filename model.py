

def Generator():
    X = Input(shape = (x_shape))#L image

  
  #C1 = ZeroPadding2D(padding=(1,1))(X)  
    C1 = Conv2D(64,kernel_size = 4, strides = 1,input_shape = x_shape,padding='same')(X)
    C1 = LeakyReLU(0.2)(C1)
#     C1=BatchNormalization()(C1)
    
#     C12 = Conv2D(64,kernel_size = 4, strides = 2,input_shape = x_shape,padding='same')(C1)
#     C12 = LeakyReLU(0.2)(C12)
#     C12=BatchNormalization()(C12)
    

    C11 = Conv2D(128,kernel_size = 4, strides = 2,padding='same')(C1)
    C11 = LeakyReLU(0.2)(C11)
    C11=BatchNormalization()(C11)
  
    C2 = Conv2D(256,kernel_size = 4, strides = 2,padding='same')(C11)  
    C2 = LeakyReLU(0.2)(C2)
    C2= BatchNormalization()(C2)


    C3 = Conv2D(512,kernel_size = 4, strides = 2,padding='same')(C2)
    C3 = LeakyReLU(0.2)(C3)
    C3= BatchNormalization()(C3)
  
    C4 = Conv2D(512,kernel_size = 4, strides = 2,padding='same')(C3)
    C4 = LeakyReLU(0.2)(C4)
    C4= BatchNormalization()(C4)
    
    C5 = Conv2D(512,kernel_size = 4, strides = 2,padding='same')(C4)
    C5 = LeakyReLU(0.2)(C5)
    C5= BatchNormalization()(C5)
    
    C6 = Conv2D(512,kernel_size = 4, strides = 2,padding='same')(C5)
    C6 = LeakyReLU(0.2)(C6)
    C6= BatchNormalization()(C6)
    
    C7 = Conv2D(512,kernel_size = 4, strides = 2,padding='same')(C6)
    C7 = LeakyReLU(0.2)(C7)
    C7= BatchNormalization()(C7)
    
    DC00 = Deconv2d(512, kernel_size = 4, strides = 2,padding='same')(C7)
    DC00 = LeakyReLU(0.2)(DC00)
    DC00 = BatchNormalization()(DC00)
    DC00 = Dropout(0.5)(DC00)
    DC00 = Concatenate(axis=3)([DC00, C6])
    
    
    DC0 = Deconv2d(512, kernel_size = 4, strides = 2,padding='same')(DC00)
    DC0 = LeakyReLU(0.2)(DC0)
    DC0 = BatchNormalization()(DC0)
    DC0 = Dropout(0.5)(DC0)
    DC0 = Concatenate(axis=3)([DC0, C5])

    DC01 = Deconv2d(512, kernel_size = 4, strides = 2,padding='same')(DC0)
    DC01 = LeakyReLU(0.2)(DC01)
    DC01 = BatchNormalization()(DC01)
    DC01 = Dropout(0.5)(DC01)
    DC01 = Concatenate(axis=3)([DC01, C4])
    
    DC02 = Deconv2d(512, kernel_size = 4, strides = 2,padding='same')(DC01)
    DC02 = LeakyReLU(0.2)(DC02)
    DC02 = BatchNormalization()(DC02)
 # DC01 = Dropout(0.5)(DC01)
    DC02 = Concatenate(axis=3)([DC02, C3])  
    
    DC03 = Deconv2d(256, kernel_size = 4, strides = 2,padding='same')(DC02)
    DC03 = LeakyReLU(0.2)(DC03)
    DC03 = BatchNormalization()(DC03)
 # DC01 = Dropout(0.5)(DC01)
    DC03 = Concatenate(axis=3)([DC03, C2])
  
    DC1 = Deconv2d(128,kernel_size=4, strides = 2,padding='same')(DC03)
    DC1 = LeakyReLU(0.2)(DC1)
    DC1 = BatchNormalization()(DC1)  
  #DC1 = Dropout(0.5)(DC1)             
    DC1 = Concatenate(axis=3)([DC1,C11])
    
#     DC22 = Deconv2d(64,kernel_size=4, strides = 2,padding='same')(DC1)
#     DC22 = LeakyReLU(0.2)(DC22)
#     DC22 = BatchNormalization()(DC22)  
#     DC22= Concatenate(axis=3)([DC22,C12])

  
    DC2 = Deconv2d(64,kernel_size=4, strides = 2,padding='same')(DC1)
    DC2 = LeakyReLU(0.2)(DC2)
    DC2 = BatchNormalization()(DC2)  
    DC2 = Concatenate(axis=3)([DC2,C1])

  
    FC = Conv2D(2,kernel_size = 1, strides = 1,activation='tanh')(DC2)#a b channels
    Y=Concatenate(axis=3)([X,FC])
  #Y output--Labimage
    m = Model(X,Y)
  #m.summary()
    return m


def Discriminator():
    X = Input(shape = x_shape)
    Y = Input(shape = y_shape)
  
    In = Concatenate(axis=3)([X,Y])
  
    C1 = Conv2D(64,kernel_size = 4, strides = 2,input_shape = x_shape)(In)
    C1 = BatchNormalization()(C1)
    C1 = LeakyReLU(0.2)(C1)
    C2 = Conv2D(128,kernel_size = 4, strides = 2)(C1)  
    C2 = BatchNormalization()(C2)
    C2 = LeakyReLU(0.2)(C2)
  
    C3 = Conv2D(256,kernel_size = 4, strides = 2)(C2)
    C3 = BatchNormalization()(C3)
    C3 = LeakyReLU(0.2)(C3)
  
    C4 = Conv2D(512,kernel_size = 4, strides = 1)(C3)
    C4 = BatchNormalization()(C4)
    C4 = LeakyReLU(0.2)(C4)
    
    C5 = Conv2D(1,kernel_size = 4, strides = 1,activation='sigmoid')(C4)
#     C5 = BatchNormalization()(C5)
    
#     D = Flatten()(C4)
#     D = Dense(128)(D)
#     D = Dense(1,activation='sigmoid')(D)
  
    model = Model([X,Y],C5)
#     opt = optimizers.Adam(lr=0.0002, beta_1=0.5)
#     model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=opt,metrics=[keras.metrics.BinaryCrossentropy(from_logits=True)])
#     model.summary()
    return model

                                                ##### MAKE INTO FUNCTION
gen_vanilla=Generator()
gen_hinge=Generator()
gen_l1=Generator()
gen_l2=Generator()
gen=Generator()
tf.keras.utils.plot_model(gen,to_file=path2+'Generator.png')

LAMBDA=100

dis=Discriminator()
tf.keras.utils.plot_model(dis,to_file=path2+'Discriminator.png')

bce_loss_object= tf.keras.losses.BinaryCrossentropy(from_logits=True)
hinge_loss_object=tf.keras.losses.Hinge()


def generator_loss(disc_generated_output, gen_output, target,flag='vanilla'):
    #Mutations we chose are vanilla bce,l1,l2,hinge
    if flag=='vanilla':
      loss = bce_loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    elif flag=='hinge':
      loss=hinge_loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    elif flag=='l2':
      loss= tf.reduce_mean(tf.square(tf.subtract(tf.ones_like(disc_generated_output), disc_generated_output)))
    elif flag=='l1':
      loss = tf.reduce_mean(tf.abs(tf.subtract(tf.ones_like(disc_generated_output), disc_generated_output)))
    else:
      print("Unknown mutation")
      return None
    #Similarilty loss to ensure output is similar to input
    l1_sim_loss = tf.reduce_mean(tf.abs(target - gen_output))

    #Combined Loss
    gen_loss=loss+(LAMBDA*l1_sim_loss)
    
    return gen_loss,l1_sim_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = bce_loss_object(tf.ones_like(disc_real_output)-0.25, disc_real_output)

    generated_loss = bce_loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = 0.5*(real_loss + generated_loss)

    return total_disc_loss
