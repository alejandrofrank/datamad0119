from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (altura, anchura, canal)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # redimensionar y convertir a escala de grises
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # Almacena en la memoria de experiencia

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# En este paso, podemos indicarle a que juego de atari va a jugar. Funciona muy bien con la mayoria.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Procedemos a hacer el modelo, usamos un modelo muy conocido diseñado por Mnih et al.
# Sacado de "Human-level control through deep reinforcement learning"
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
if K.image_dim_ordering() == 'tf':
    # (anchura, altura, canal) v
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (canal, anchura, altura)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finalmente, configuramos y compilamos nuestro agente. Puede utilizar cada optimizador incorporado 
# de Keras e Incluso las métricas.
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Seleccionamos el motor. Utilizamos la selección de eps-greedy, Utilizamos el proceso de annealing
# a eps de 1.0 a 0.1 en el transcurso de pasos de 1M. Esto se hace para que el agente explora 
# inicialmente el entorno (alta eps) y luego se adhiere gradualmente a lo que sabe
# (bajo eps). También establecemos un valor de eps dedicado que se utiliza durante las pruebas. 
# Tenga en cuenta que lo configuramos a 0.05 para que el agente aún realice algunas acciones 
# aleatorias. Esto asegura que el agente no puede atascarse.

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# El intercambio entre la exploración y la explotación es difícil y un tema de investigación en curso.
# Usare para calentar el modelo unos 50000 steps antes de entrenarlo. Cuando llegue a los 10000 steps,
# se guardara un checkpoint por cada 10000.

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':

    # En este paso, se dara el entrenamiento del modelo. 
    # Capturamos la excepción de interrupción para que el entrenamiento pueda ser abortado  
    # prematuramente. ¡Observe que ahora puede utilizar las devoluciones de 
    # llamada de Keras incorporadas!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=100000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

    # Cuando el entrenamiento termina, guardamos los pesos del modelo.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finalmente, testeamos el algoritmo por un numero de episodios igual a 10.

    # Decidimos visualizar el test, es un poco mas rapido si no lo visualizamos.
    dqn.test(env, nb_episodes=10, visualize=True)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)

