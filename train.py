import time
import datetime

gamma=0.001
def fitness(grads,preds):
  div_fit=tf.multiply(gamma,tf.math.log(sum(tf.reduce_sum(tf.square(x)) for x in grads)))

  qual_fit=tf.reduce_mean(preds)
  fitness=div_fit+qual_fit
  return fitness


log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))






@tf.function
def train_step(inputs, targets,batch): 
    #Train Discriminator
    with tf.GradientTape() as disc_tape:
      outputs = gen(inputs , training=True)
      disc_real_output = dis([inputs, targets], training=True)
      disc_generated_output = dis([inputs,outputs], training=True)
      disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
    discriminator_gradients = disc_tape.gradient(disc_loss,dis.trainable_variables)
    optimizer_d.apply_gradients(zip(discriminator_gradients,dis.trainable_variables))
    del outputs,disc_real_output,disc_generated_output,discriminator_gradients
  #Train individual Mutations
    #Vanilla
    with tf.GradientTape() as v_tape:
        outputs_v = gen_vanilla(inputs , training=True)
        disc_real_output_v = dis([inputs, targets], training=True)
        disc_generated_output_v = dis([inputs,outputs_v], training=True)
        v_loss,v_sim_loss=generator_loss(disc_generated_output_v,outputs_v,targets,flag='vanilla')
    
    vanilla_generator_gradients = v_tape.gradient(v_loss,gen_vanilla.trainable_variables)
    optimizer_v_g.apply_gradients(zip(vanilla_generator_gradients,gen_vanilla.trainable_variables))
    del outputs_v,disc_real_output_v,disc_generated_output_v,v_loss,v_sim_loss,vanilla_generator_gradients


    #Hinge
    with tf.GradientTape() as h_tape:
        outputs_h = gen_hinge(inputs , training=True)
        disc_real_output_h = dis([inputs, targets], training=True)
        disc_generated_output_h = dis([inputs,outputs_h], training=True)
        h_loss,h_sim_loss=generator_loss(disc_generated_output_h,outputs_h,targets,flag='hinge')
    
    hinge_generator_gradients = h_tape.gradient(h_loss,gen_hinge.trainable_variables)
    optimizer_h_g.apply_gradients(zip(hinge_generator_gradients,gen_hinge.trainable_variables))
    del outputs_h,disc_real_output_h,disc_generated_output_h,h_loss,h_sim_loss,hinge_generator_gradients


    #L1
    with tf.GradientTape() as l1_tape:
        outputs_l1 = gen_l1(inputs , training=True)
        disc_real_output_l1 = dis([inputs, targets], training=True)
        disc_generated_output_l1 = dis([inputs,outputs_l1], training=True)
        l1_loss,l1_sim_loss=generator_loss(disc_generated_output_l1,outputs_l1,targets,flag='l1')
    
    l1_generator_gradients = l1_tape.gradient(l1_loss,gen_l1.trainable_variables)
    optimizer_l1_g.apply_gradients(zip(l1_generator_gradients,gen_l1.trainable_variables))
    del outputs_l1,disc_real_output_l1,disc_generated_output_l1,l1_loss,l1_sim_loss,l1_generator_gradients

    #L2
    with tf.GradientTape() as l2_tape:
        outputs_l2 = gen_l2(inputs , training=True)
        disc_real_output_l2 = dis([inputs, targets], training=True)
        disc_generated_output_l2 = dis([inputs,outputs_l2], training=True)
        l2_loss,l2_sim_loss=generator_loss(disc_generated_output_l2,outputs_l2,targets,flag='l2')
    
    l2_generator_gradients = l2_tape.gradient(l2_loss,gen_l2.trainable_variables)
    optimizer_l2_g.apply_gradients(zip(l2_generator_gradients,gen_l2.trainable_variables))
    del outputs_l2,disc_real_output_l2,disc_generated_output_l2,l2_loss,l2_sim_loss,l2_generator_gradients

 #After training we evaluate the fitness of each mutation:
    
     #Vanilla
    with tf.GradientTape() as v_v_tape:
        outputs_v_v = gen_vanilla(inputs , training=True)
        disc_real_output_v_v = dis([inputs, targets], training=True)
        disc_generated_output_v_v = dis([inputs,outputs_v_v], training=True)
        v_loss_v_v,v_sim_loss_v_v=generator_loss(disc_generated_output_v_v,outputs_v_v,targets,flag='vanilla')
    
    vanilla_generator_gradients_v_v = v_v_tape.gradient(v_loss_v_v,gen_vanilla.trainable_variables) 
    vanilla_fit=fitness(vanilla_generator_gradients_v_v,disc_generated_output_v_v)
    del outputs_v_v,disc_real_output_v_v,disc_generated_output_v_v,v_loss_v_v,v_sim_loss_v_v,vanilla_generator_gradients_v_v

    #Hinge
    with tf.GradientTape() as h_h_tape:
        outputs_h_h = gen_hinge(inputs , training=True)
        disc_real_output_h_h = dis([inputs, targets], training=True)
        disc_generated_output_h_h = dis([inputs,outputs_h_h], training=True)
        h_loss_h_h,h_sim_loss_h_h=generator_loss(disc_generated_output_h_h,outputs_h_h,targets,flag='hinge')
    
    hinge_generator_gradients_h_h = h_h_tape.gradient(h_loss_h_h,gen_hinge.trainable_variables)
    hinge_fit=fitness(hinge_generator_gradients_h_h,disc_generated_output_h_h)
    del outputs_h_h,disc_real_output_h_h,disc_generated_output_h_h,h_loss_h_h,h_sim_loss_h_h,hinge_generator_gradients_h_h

    #L1
    with tf.GradientTape() as l1_l1_tape:
        outputs_l1_l1 = gen_l1(inputs , training=True)
        disc_real_output_l1_l1 = dis([inputs, targets], training=True)
        disc_generated_output_l1_l1 = dis([inputs,outputs_l1_l1], training=True)
        l1_loss_l1_l1,l1_sim_loss_l1_l1=generator_loss(disc_generated_output_l1_l1,outputs_l1_l1,targets,flag='l1')
    
    l1_generator_gradients_l1_l1 = l1_l1_tape.gradient(l1_loss_l1_l1,gen_l1.trainable_variables)
    l1_fit=fitness(l1_generator_gradients_l1_l1,disc_generated_output_l1_l1)
    del outputs_l1_l1,disc_real_output_l1_l1,disc_generated_output_l1_l1,l1_loss_l1_l1,l1_sim_loss_l1_l1,l1_generator_gradients_l1_l1


    #L2
    with tf.GradientTape() as l2_l2_tape:
        outputs_l2_l2 = gen_l2(inputs , training=True)
        disc_real_output_l2_l2 = dis([inputs, targets], training=True)
        disc_generated_output_l2_l2 = dis([inputs,outputs_l2_l2], training=True)
        l2_loss_l2_l2,l2_sim_loss_l2_l2=generator_loss(disc_generated_output_l2_l2,outputs_l2_l2,targets,flag='l2')
    
    l2_generator_gradients_l2_l2 = l2_l2_tape.gradient(l2_loss_l2_l2,gen_l2.trainable_variables)
    l2_fit=fitness(l2_generator_gradients_l2_l2,disc_generated_output_l2_l2)
    del outputs_l2_l2,disc_real_output_l2_l2,disc_generated_output_l2_l2,l2_loss_l2_l2,l2_sim_loss_l2_l2,l2_generator_gradients_l2_l2

    with summary_writer.as_default():
      tf.summary.scalar('vanilla_fitness', vanilla_fit, step=batch)
      tf.summary.scalar('hinge_fitness', hinge_fit, step=batch)
      tf.summary.scalar('l1_fitness', l1_fit, step=batch)
      tf.summary.scalar('l2_fitness', l2_fit, step=batch)
      tf.summary.scalar('disc_loss',disc_loss,step=batch)



    
    # if tf.equal(fitt,vanilla_fit):
    #   return 'v'
    # elif tf.equal(fitt,hinge_fit):
    #   return 'h'
    # elif tf.equal(fitt,l1_fit):
    #   return 'l1'
    # elif tf.equal(fitt,l2_fit):
    #   return 'l2'
    # else:
    #   return 'none'
    return vanilla_fit,hinge_fit,l1_fit,l2_fit
    # del gen_tape,generator_gradients,discriminator_gradients
    # del disc_tape,gen_total_loss, gen_gan_loss, gen_l1_loss,disc_loss

def fit(train_ds, epochs):
  for epoch in range(epochs):
    start = time.time()

    # display.clear_output(wait=True)

    # for example_input, example_target in test_ds.take(1):
    #   generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)
    btc=0
    # Train
    # with tf.profiler.experimental.Trace('train', step_num=1, _r=1):

    for n, (input_image, target) in train_ds.enumerate():
      print('.',btc,'-', end='')
      if (n+1)%10==0:
        gen.save(path2+'egen.h5')
        dis.save(path2+'edis.h5')
        print('--Saved--',end='')
      if (n+1) % 100 == 0:
        print()
        printm()
        print()
      all_fit=train_step(input_image, target, n)
      max_fit=max(all_fit)
      if max_fit==all_fit[0]:
        flag='v'
      elif max_fit==all_fit[1]:
        flag='h'
      elif max_fit==all_fit[2]:
        flag='l1'
      elif max_fit==all_fit[3]:
        flag='l2'


      if flag=='v':
        gen.set_weights(gen_vanilla.get_weights())
        gen_hinge.set_weights(gen_vanilla.get_weights())
        gen_l1.set_weights(gen_vanilla.get_weights())
        gen_l2.set_weights(gen_vanilla.get_weights())
      elif flag=='h':
        gen.set_weights(gen_hinge.get_weights())
        gen_vanilla.set_weights(gen_hinge.get_weights())
        gen_l1.set_weights(gen_hinge.get_weights())
        gen_l2.set_weights(gen_hinge.get_weights())
      elif flag=='l1':
        gen.set_weights(gen_l1.get_weights())
        gen_hinge.set_weights(gen_l1.get_weights())
        gen_vanilla.set_weights(gen_l1.get_weights())
        gen_l2.set_weights(gen_l1.get_weights())
      elif flag=='l2':
        gen.set_weights(gen_l2.get_weights())
        gen_hinge.set_weights(gen_l2.get_weights())
        gen_l1.set_weights(gen_l2.get_weights())
        gen_vanilla.set_weights(gen_l2.get_weights())
      print(flag,'.',end='')

      if (n+1)%500==0:
        clear_output(wait=True)
        # show_predictions(1)
      btc+=1


    print()

    # saving (checkpoint) the model every 20 epochs
    # if (epoch + 1) % 20 == 0:
    #   checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  # checkpoint.save(file_prefix = checkpoint_prefix)

                                                     ### Tensorboard stuff
%load_ext tensorboard
%tensorboard --logdir {log_dir} 

fit(train_data,10)

gen.save(path2+'gen_model.h5')
dis.save(path2+'dis_model.h5')



import cv2
def display(display_list):
    plt.figure(figsize=(15, 15))
    # for x in display_list:
        # print(type(display_list))

    title = ['Input Image', 'True', 'Predicted']

    for i in range(len(display_list)):
        if display_list[i].shape[2]==1:
            display_list[i]=cv2.cvtColor(display_list[i], cv2.COLOR_GRAY2BGR)
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


import random
def show_predictions(dataset=None, num=1):
    if dataset:
        tr=train_data.shuffle(30)
        for n,(xx,yy) in tr.enumerate():
#             x,y=xx[0],yy[0]
#             print(xx.shape,yy.shape,type(x),type(y))

#             li=np.reshape(x,(1,256,256))
            pred_mask = gen(xx,training=False)
#             pred=gen(x,training=False)
            disc_generated_output = dis([xx,pred_mask], training=False)
            disc_real_output = dis([xx,yy], training=False)
            print('discriminator prediction for generated image-->',tf.keras.backend.get_value(tf.math.reduce_mean(disc_generated_output)),"\ndiscriminator prediction for real image",tf.keras.backend.get_value(tf.math.reduce_mean(disc_real_output)))
            #       print(pred_mask.shape)
            x=xx.numpy()
            y=yy.numpy()
            pred=pred_mask.numpy()   
            nnn = random.randint(0,30)
# print(type(x),type(y),type(pred_mask))

            display([x[nnn],y[nnn],pred[nnn]])
            if n%10==0:
                break
    else:
        print("ERRROROROROOR")

import time
for i in range(20):    
    # clear_output(wait=True)
    show_predictions(1)
    time.sleep(1)
