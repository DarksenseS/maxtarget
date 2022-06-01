# Базовые Модули
import time    # модуль для операций со временными характеристиками
import random
import numpy as np

# Модули Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam

# Модули Keras-RL2
import rl.core as krl
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

# Модули визуализации
from celluloid import Camera
import matplotlib.pyplot as plt     
from matplotlib import rc
rc('animation', html='jshtml')
%matplotlib inline

# Имитация роевого поведения
class Colony:
  # положения частицы 
  x : np.ndarray
  y : np.ndarray
  # угол направления частицы
  theta : np.ndarray
  # скорость частицы по осям
  vx : np.ndarray
  vy : np.ndarray

  # Конструктор
  def __init__(self,N):
    self.reset(N)

  # расстановка N частиц на площадке LxL
  def reset(self,N):
    # положения частиц 
    self.x = np.random.rand(N,1)*L
    self.y = np.random.rand(N,1)*L
    # направление и осевые скорости частиц относительно 
    # постоянной линейной скорости v0
    self.theta = 2 * np.pi * np.random.rand(N,1)
    self.vx = v0 * np.cos(self.theta)
    self.vy = v0 * np.sin(self.theta)
  # Шаг имитации
  def step(self):
    # движение
    self.x += self.vx*dt
    self.y += self.vy*dt
    # применение периодических пограничных условий
    self.x = self.x % L
    self.y = self.y % L
    # найти средний угол соседей в диапазоне R
    mean_theta = self.theta
    for b in range(N):
        neighbors = (self.x-self.x[b])**2+(self.y-self.y[b])**2 < R**2
        sx = np.sum(np.cos(self.theta[neighbors]))
        sy = np.sum(np.sin(self.theta[neighbors]))
        mean_theta[b] = np.arctan2(sy, sx)
    # добавление случайного отклонения
    self.theta = mean_theta + eta*(np.random.rand(N,1)-0.5)
    # изменение скорости
    self.vx = v0 * np.cos(self.theta)
    self.vy = v0 * np.sin(self.theta)
    return self.theta

  # Получить список частиц в внутри радиуса r от координат x,y
  def observe(self,x,y,r):
    return (self.x-x)**2+(self.y-y)**2 < r**2
  # Вывести координаты частицы i
  def print(self,i):
    return print(self.x[i],self.y[i])
  # Получить координаты частиц
  def get_fishi(self):
    return self.x, self.y 
  # Получить массив направлений частиц
  def get_theta(self):
    return self.theta
    # action - скаляр от -1 до 1
class actionSpace(krl.Space):
  def __init__(self):
    self.shape = (1,)
  def sample(self, seed=None):
    if seed: random.seed(seed)
    return random.triangular(-1,1)
  def contains(self, x):
    return  abs(x) <= 1

# observation - массив 
# допустимые значения можно не описывать.
class observationSpace(krl.Space):
  def __init__(self):
    self.shape = (5,) #
  def sample(self, seed=None): pass
  def contains(self, x): pass
  class Cure(krl.Env):
  # имитируемая колония
  fishi : Colony
  # положение нано робота
  x: float
  y: float
  theta: float  # направление нано робота
  R: float  # область видимости рыб нано роботом
  n_fishi : int  # сохраняем предыдущее значение количества видимых рыб для rewarda
  # конструктор
  def __init__(self):
    self.fishi = Colony(N)
    self.reward_range = (-1,1) #(-np.inf, np.inf)
    self.action_space = actionSpace()
    self.observation_space = observationSpace()
    self.R = observation_R
    self.reset()

  #  Формирование вектора обзора observation.
  #  То что происходит в области видимости R от робота. 
  def observe_area(self):
    # получим список соседей в радиусе R
    observe_fishi = self.fishi.observe(self.x,self.y,self.R)
    # получим список соседей в радиусе R*1.5
    observe_far_fishi = self.fishi.observe(self.x,self.y,self.R*1.5)
    observe_far_fishi=np.array(np.bitwise_and(observe_far_fishi,np.invert (observe_fishi)))

    observation = np.zeros(5)
    # подадим количество соседей    
    n_fishi = np.sum(observe_fishi)
    observation[0] = n_fishi/20

    # посчитаем и подадим среднее направлений соседних рыб
    sx = np.sum(np.cos(self.fishi.theta[observe_fishi]))
    sy = np.sum(np.sin(self.fishi.theta[observe_fishi]))
    observation[1] = np.arctan2(sy, sx)/np.pi
    # посчитаем и подадим среднее направление от робота до удаленных рыб
    sx = np.sum(self.fishi.x[observe_fishi]-self.x)
    sy = np.sum(self.fishi.y[observe_fishi]-self.y)
    observation[2] = np.arctan2(sy, sx)/np.pi
    # посчитаем и подадим среднее направление от робота до удаленных рыб
    sx = np.sum(self.fishi.x[observe_far_fishi]-self.x)
    sy = np.sum(self.fishi.y[observe_far_fishi]-self.y)
    observation[3] = np.arctan2(sy, sx)/np.pi
    if n_fishi:
      observation[4]=self.theta/np.pi # подадим направление наноробота
    return np.sum(observe_fishi), observation

  # старт симуляции
  def reset(self):
    self.fishi.reset(N)
    self.x = .5*L
    self.y = .5*L
    self.theta = actionSpace().sample()
    self.n_fishi , observation = self.observe_area()
    return observation
    
  # шаг симуляции
  def step(self,action):
    action = action * 3.2#np.pi
    #  Для экономии времени при попадании на "чистую воду" 
    #  просчитываем симуляцию не выпуская ее для обработки сети
    while True:
      # шаг симуляции рыб
      self.fishi.step()
      # шаг робота
      self.theta = np.sum(action) #% (2*np.pi)
      self.x = self.x + dt*v0 * np.cos(self.theta)
      self.y = self.y + dt*v0 * np.sin(self.theta)
      self.x = self.x  % L
      self.y = self.y  % L
      # осматриваем окружение
      nfishi, observation = self.observe_area()
      if np.sum(observation)!=0: break
      if self.n_fishi > 0: break

    delta = nfishi - self.n_fishi
    if delta<0:
      reward = 50 * delta/self.n_fishi
    elif delta>0 and self.n_fishi:
      reward = 1+delta
    elif nfishi>0:
      reward = 1        
    elif nfishi == 0:
      reward = 0
    else: 
      reward = nfishi
    done = nfishi > N/7
    self.n_fishi = nfishi
    return observation, reward, done, {}

  # получить координаты робота
  def get_position(self):
    return self.x, self.y, self.R
  # получить координаты всех рыб
  def get_fishi(self):
    return self.fishi.get_fishi()
  # отразить отладочную информацию   
  def render(self, mode='human', close=False):
    #print(self.n_fishi)
    pass
  # завершить симуляцию
  def close(self): pass
  
# Создадим среду и извлечем пространство действий
env = Cure()
np.random.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Построим модель актера. Подаем среду, получаем действие
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(4, use_bias=True))
actor.add(Activation('relu'))
actor.add(Dense(4, use_bias=True))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions, use_bias=True))
actor.add(Activation('tanh'))
print(actor.summary())

# Построим модель критика. Подаем среду и действие, получаем награду
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(8, use_bias=False)(x)
x = Activation('relu')(x)
x = Dense(5, use_bias=True)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Keras-RL предоставляет нам класс, rl.memory.SequentialMemory
# где хранится "опыт" агента:
memory = SequentialMemory(limit=100000, window_length=1)
# чтобы не застрять с локальном минимуме, действия модели полезно "встряхивать" случайным поведением 
# с помощью Процесса Орнштейна – Уленбека
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
# Создаем agent из класса DDPGAgent
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)

agent.compile(Adam(learning_rate=.001, clipnorm=1.), metrics=['mae'])

# Обучим процесс на nb_steps шагах, 
# nb_max_episode_steps ограничивает количество шагов в одном эпизоде
agent.fit(env, nb_steps=100000, visualize=True, verbose=1, nb_max_episode_steps=Epochs)

# Тестируем обученую сеть на 5 эпизодах
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=Epochs)
env.close()
v0 = 4        # линейная скорость
N = 1000      # количество рыб
Epochs =  500 # количество шагов
L    = 300    # размер области
R    = 5      # радиус взаимодействия
observation_R = 2*R # Радиус видимости соседей

fig = plt.figure()
camera = Camera(fig)
random.seed(123)
theCure = Cure()
observation = theCure.reset()

# информационная плашка
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
sum_reward = 0
for i in range(200):
    action = np.sum(actor.predict(observation.reshape((1,1,5))))# % (2*np.pi)
    observation, reward, done, _ = theCure.step(action)
    sum_reward += reward
    if done:
      print('Победа  на шаге',i, ' захвачено ',observation[0]*20,'рыб. Награда ',sum_reward)
      break
    fishi_x,fishi_y = theCure.get_fishi()
    plt.scatter(fishi_x, fishi_y, c='red')    #  метод, отображающий данные в виде точек
    # покажем робота
    x, y, r = theCure.get_position()
    plt.scatter(x, y, c='blue')
    fig = plt.gcf()
    ax = fig.gca()
    circle = plt.Circle((x, y), r, color='b', fill=False)
    ax.add_patch(circle)

    textstr = '\n'.join((
    r'epoch=%d' % (i, ),
    r'points=%d' % (reward, ),
    ))

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
      verticalalignment='top', bbox=props)

    camera.snap()

print('Итоговое вознаграждение',sum_reward)
theCure.close()
animation = camera.animate()
#animation.save('celluloid_minimal.gif', writer = 'imagemagick')
animation
