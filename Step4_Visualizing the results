import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env):
  state, _ = env.reset()
  done = False
  frames = []
  while not done:
    frame = env.render()
    frames.append(frame)
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action[0])
  env.close()
  imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, env)

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()
