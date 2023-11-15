# vast.ai-guide
A guide on how to navigate and use Vast.ai for machine learning. For use case, we will use [RAVE](https://github.com/acids-ircam/RAVE). [Here is the FAQ for Vast.ai](https://vast.ai/faq), but I will explain some practical things here. 

***Vast lets you connect to other peoples computers who rents out their GPU through Vast. Most of the how-to-steps here are provided from the Vast website. Take extra care when connecting Drives and uplaoding passwords. I'm no computer security expert, so please use this guide at your own risk.***

Head over to the vast.ai website. Create an account. Click on "billing" and buy some credits. We will need this in order to rent a computer. 

### Choosing an instance
Go to "templates". There are computers that you can use with a simple use interface where you can see a tree like structure of files, with upload and download buttons, as well as options to open up a terminal window, or to create a new notebook to run code. If you like this option, click the "Pytorch... something". I usually scroll down and choose the most popular one. However, for this guide we will connect to the a computer using SSH and run everything in the CLI (command line interface, ie. a terminal window). Scroll down to most popular and choose the "Cuda:<version>Devel-Ubuntu20.04" by clicking on "select". 

### Rent computer
You will be taken to a new site with prices on the right. I usually use 1xTRX 4090", that offers great performance. Next, there are some options to the left you might want to edit. For RAVE training, you won't need more than some 20-30 gigs probably, but that depens on the size of your dataset, and how many versions you decide to train. Also notice the "max instance duration" parameter, adjust it to your needs. Lastly, pay extra attention to the "Secure Cloud" box. Click this if you intend to connect your drive. By clicking this, you will filter out computers that are not Vasts own computers. More on this later! 

### Connect to computer 
After you have chosen a computer, click rent, and head over to "instances" (see tab on the left). There you will see your instance. I usually connect with SSH, because this enables you to run processes in the background. That means that training sessions are immune to interuptions, and will keep on running. This is not possible when running a notebook. More on this later. To connect via SSH, [see these steps](https://vast.ai/faq#how-do-i-connect-to-an-ssh-instance-on-linuxmac). Basically, generate a key on your PC, grab API from Vast website, paste it into your terminal, and connect using the command that pops up when you click "Connect" on one of your instances. If you choose to connect just using the browser, you will get warnings that the site is not secure. I'm guessing this is because you're connecting to someone elses computer. [More about that here](https://vast.ai/faq#what-is-this-https-website-unsecure-warning).

### Transfer files
Now that we're connected, let's see how to transfer files. As far as I know, there are two ways to do it: 

**Cloud sync**. Look for the little cloud-symbol underneath the rent button. Click that if you want to connect a drive. Here you can choose if you want to migrate files from cloud to instance, or the other way around (click on the "migrate to..." blue text, and it will change mode). However, we need a cloud integration! [See this guide to get this set up](https://vast.ai/docs/gpu-instances/cloud-sync). Note that you might have to repeat these steps whenever you sync a new instance. 

**SSH**. With a little modification, this is quite easy, and personally my favorite way to move files. [Here is a Vast guide with the steps](https://vast.ai/faq#data-movement). In the example provided, they use this code: 
```
./vast copy ~/workspace 4330147:/workspace
```
However, we first need to download vastai. Run the following command on your local computer:
```
pip install vastai
```
Alternatively: 
```
wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;
```

(If you don't have pip, look further down in this readme where we install it for Linux. When using the vastai, you might get an error, like, "don't recognize pip, no such command", that means that vastai is not in your so called "PATH". You can fix this by adding vastai to you PATH. We will also go through this a little later, but for Linux, although the method is the same). After you have installed vastai on your computer, you can run the following command to copy files: 
```
vastai copy ~/workspace 4330147:/workspace
```
This will copy files from the folder "workspace" in your homefolder. If you are unfamiliar with paths in your terminal, use the cd command to cd to the folder you want to transfer, and type "pwd", and you will see the full path. The number "4330147" is the number of your instance. Head over to the instances tab, and you'll find a number to the left on your instance. Grab the number and put it into the copy command. In the code above, files are copied from your local computer to the remote instance, to do it the other way around, simple change the order around, for example: 
```
vastai copy 4330147:/workspace ~/workspace
```
It will probably ask you for a password. Again, I'm no computer security expert, I don't know how safe this is. 

### Install miniconda, ffmpeg and rave
Let's start installing some software on our remote instance so that we can eventually start a training. Some of these steps are from [hexorcismos](https://github.com/moiseshorta) google colab notebook. [Link to that here](https://colab.research.google.com/drive/1ih-gv1iHEZNuGhHPvCHrleLNXvooQMvI?usp=sharing). The code below downloads a .sh file using the curl command, then changes the permissions so that we can run it with the sh command. The semicolon separates the different commands.
**Install miniconda** 
```
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh; chmod +x; miniconda.sh; sh miniconda.sh -b -p /content/miniconda
```
**Install rave and upgrade ipython**
```
/content/miniconda/bin/pip install acids-rave; /content/miniconda/bin/pip install --upgrade ipython ipykernel
```
**Install ffmpeg**
```
/content/miniconda/bin/conda install ffmpeg --yes
```

**Add everything to the PATH**
In order to avoid errors when training, we need to add ffmpeg to the PATH. We can do this the easy way with a single command:
```
export PATH="/content/miniconda/bin:$PATH"
```
However, with this command, only one terminal session will have be able to use the path. It's sort of like a temporary fix, and we would have to run it every time we log out and back in. As we will see later on, we will need more than one session window, so let's go ahead and add it in the PATH another way. For this we will need a text editor. My favorite choise of editor is "nano". Let's install it with the following command: 
```
sudo apt-get install nano; nano ~/.bashrc
```
The last part opens up the bash profile. Scroll down until a piece of text looking something like this: 
```
PATH='/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
```
Edit the text (use the arrows to navigate) and add "/content/miniconda/bin" to it, like this: 
```
PATH='/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/content/miniconda/bin'
```
Exit by clicking ctrl + x, save it by pressing "y". In order for our new PATH to work, we need to refresh it, we do this by typing:
```
source ~/.bashrc
```
**Edit dataset.py file**
While were out and about with our new nano editor, let's do some more edits! There is a file called dataset.py, which throws an error (something like "AttributeError: module 'numpy' has no attribute 'float'") if we don't change replace "np.float" with "float". Nano is the perfect tool for this. So open up the file by typing: 
```
nano /content/miniconda/lib/python3.11/site-packages/rave/dataset.py
```
and navigate down some 60 lines or so until you see the line *audio = audio.astype(np.float) / (2**15 - 1)"*, and change "np.float" to just "float". Exit with ctrl + c, and save with y.

### Preprocess
Now follow the steps for preprocessing on the [git repo](https://github.com/acids-ircam/RAVE). After running your first preprocess command, you  notice that the terminal is acting weird, like if it froze. No worries. The terminal is still running, but no letters are showing up. Fix this by typing: 
```
reset
```
and hit enter. Notice, you won't actually see the text "reset" as you type it. 

### Training, how to monitor a background process and how to stop a training
Let's start training! Again, use the commands on the git repo to start training. If you managed to get the training to start without any type-o's, pointing to wrong folders etc, you should see it going through epochs by now. Great! Let's say you want to leave it going for a some time, then you might want to run it as a background process. This was oiKys idea, [see discord post here](https://discord.com/channels/987249093124452400/987249554745356298/1168904219575726291). Stop your training with ctrl + c. To run it as a background process, add "nohup" to the beginning of the code, and "&" at the end (there  is a way to move it to the background without interupting the training, but I don't know how). For example: 
```
nohup rave train --config some-configuration --db_path /dataset/path --name give_a_name &
```
All the logs from the training are going out to a document called "nohup.out". But how can we see that document while the nohup training is running? We need another window! Follow [these steps](https://vast.ai/faq#what-is-this-tmux-thing-how-do-i-create-multiple-bash-terminals-on-my-ssh-instance). Simply press ctrl + b, release and click c (c as in create). To toggle between the original and the newly created window, press ctrl + b followed by n (n as in next I guess). Create as many windows as you like! (Spoiler alert, we will create one to run a Tensorboard monitor later!)

In the newly created window, navigate to where you were when you started training using the cd command. Google cd command if you are not familiar with it. In this folder you will find file called "nohup.out". Open it, and view realtime incoming data by typing: 
```
tail -f nohup.out
```
You should see the training going through iterations now. 

### Tensorboard monitoring
How can we run a tensorboard monitor in the terminal, it's got no user interface? No problem, we will run it in the localhost! But how can we monitor a localhost when we're renting a computer from a remote server? No problem, we will route localhost server through the ssh hole. This is much easier than it sounds. First, we begin by installing Tensorboard on our remote instance: 
```
pip install tensorboard
```
Start the tensorboard from the home folder of your training (the folder where you are when you run the "rave train" command) by cd'ing there, for instance: 
```
cd /path/to/rave/folder
```
Now, start the tensorboard: 
```
tensorboard --logdir . --port=8080
```
Notice the port number 8080? It's the same port number we used to connect! When we connect using SSH, we see something like: 
```
ssh -p 7417 root@52.204.230.7 -L 8080:localhost:8080
```
This is why we start the tensorboard on that same port. Not knowing too much about this port magic, lets open up the browser on our local computer and enter the following adress: 
```
localhost:8080
```
Now, you should see the tensorboard loading! Whenever you want see the latest updates from the training, simply refresh the localhost website. 

### Resuming from checkpoints and exporting
Enter the code from the RAVE repo. The checkpoints are in the "runs" folder. Navigate using the cd command. Hint, double tapping the tab key will display all files in the folder you're in, making it simple to navigate around. Use "cd .." to go up one level. If you have multiple runs, navigate to the latest one. Use best checkpoint or last checkpoint? I don't know. 

### Prior
Follow steps from the prior website. One that that really confused me with training a prior, was that you're supposed to point to a .ts file that's generated in the pre-processed folder - NOT the .ts file that's generated when running "rave export". 

### Ending words
I hope you find this helpful! 
