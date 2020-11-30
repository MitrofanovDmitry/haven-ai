import sys, os, pprint

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

# from haven.haven_jobs import slurm_manager as sm

  
from haven import haven_chk as hc
# from haven import haven_results as hr
from haven import haven_utils as hu
from haven import haven_examples as he 
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import time
import numpy as np
from torch.utils.data import RandomSampler, DataLoader
print()

try:
  import job_configs
except:
  pass
# from haven_utils import file_utils




# Job status
# ===========
def get_job(job_id):
    """Get job information."""
    command = "scontrol show job %s" % job_id
    job_info = {}
    while True:
      try:
        job_info = hu.subprocess_call(command)
        job_info = job_info.replace('\n', '')
        job_info = {v.split('=')[0]:v.split('=')[1] for v in job_info.split(' ') if '=' in v }
      except Exception as e:
        if "Socket timed out" in str(e):
          print("scontrol time out and retry now")
          time.sleep(1)
          continue
        else:
          job_info = {'JobState': 'Invalid Job Id'}
      break
    return job_info
    

def get_jobs(user_name):
    # account_id = hu.subprocess_call('eai account get').split('\n')[-2].split(' ')[0]
    ''' get the first 3 jobs'''
    command = "squeue --user=%s" % user_name
    while True:
      try:
        job_list = hu.subprocess_call(command)
        job_list = job_list.split('\n')
        job_list = [v.lstrip().split(" ")[0] for v in job_list[1:]]
        result = []
        for i in range(0, 3):
          result.append(get_job(job_list[i]))
      except Exception as e:
        if "Socket timed out" in str(e):
          print("scontrol time out and retry now")
          time.sleep(1)
          continue
      break
    return result

           
# Job kill
# ===========
def kill_job(job_id):
    """Kill a job job until it is dead."""
    kill_command  = "scancel %s" % job_id
    while True:
      try:
        hu.subprocess_call(kill_command)     # no return message after scancel
      except Exception as e:
        if "Socket timed out" in str(e):
          print("scancel time out and retry now")
          time.sleep(1)
          continue
      break
    return
import argparse

def get_job_id(exp_id, savedir_base):
  savedir = os.path.join(savedir_base, exp_id)
  file_name = os.path.join(savedir, "job_dict.json")
  if not os.path.exists(file_name):
    return -1
  job_dict = hu.load_json(file_name)
  return job_dict["job_id"]
  

def get_existing_slurm_job_commands(exp_list, savedir_base):
    existing_job_commands = []
    for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        file_name = os.path.join(savedir_base, exp_id, "job_dict.json")
        if not os.path.exists(file_name):
            continue
        job_dict = hu.load_json(file_name)
        job_id = job_dict["job_id"]
        job_status = hu.subprocess_call("scontrol show job %s" % job_id).split("JobState=")[1].split(" ")[0]
        if job_status == "RUNNING" or job_status == "PENDING":
            existing_job_commands += [job_dict["command"]]
        
    return existing_job_commands

def get_job_spec():
    # read slurm setting
    lines = "#! /bin/bash \n"
    lines += "#SBATCH --account=%s \n" % job_configs.ACCOUNT_ID
    for key in list(job_configs.JOB_CONFIG.keys()):
        lines += "#SBATCH --%s=%s \n" % (key, job_configs.JOB_CONFIG[key])
    return lines

def submit_job(command, savedir):
    # read slurm setting
    lines = "#! /bin/bash \n"
    # if job_config is not None:
    #     lines += "#SBATCH --account=%s \n" % job_configs.ACCOUNT_ID
    #     for key in list(job_config.keys()):
    #         lines += "#SBATCH --%s=%s \n" % (key, job_config[key])
    lines += "#SBATCH --account=%s \n" % job_configs.ACCOUNT_ID
    for key in list(job_configs.JOB_CONFIG.keys()):
      lines += "#SBATCH --%s=%s \n" % (key, job_configs.JOB_CONFIG[key])
    path_log = os.path.join(savedir, "logs.txt")
    path_err = os.path.join(savedir, "err.txt")
    lines += "#SBATCH --output=%s \n" % path_log
    lines += "#SBATCH --error=%s \n" % path_err
    # path_code = os.path.join(savedir, "code")
    # lines += "#SBATCH --chdir=%s \n" % path_code
    
    lines += command

    file_name = os.path.join(savedir, "bash.sh")
    hu.save_txt(file_name, lines)
    # launch the exp
    submit_command = "sbatch %s" % file_name
    while True:
      try:
        job_id = hu.subprocess_call(submit_command).split()[-1]
      except Exception as e:
        if "Socket timed out" in str(e):
          print("slurm time out and retry now")
          time.sleep(1)
          continue
      break

    # save the command and job id in job_dict.json
    job_dict = {
        "command": command,
        "job_id": job_id
    }
    hu.save_json(os.path.join(savedir, "job_dict.json"), job_dict)

    # delete the bash.sh
    os.remove(file_name)

    return job_id


def save_exp_folder(exp_dict, savedir_base, reset):
    exp_id = hu.hash_dict(exp_dict)  # generate a unique id for the experiment
    savedir = os.path.join(savedir_base, exp_id)  # generate a route with the experiment id
    if reset:
        hc.delete_and_backup_experiment(savedir)

    os.makedirs(savedir, exist_ok=True)  # create the route to keep the experiment result
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)  # save the experiment config as json


from haven import haven_wizard as hw
# 1. define the training and validation function
def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # 2. Create data loader and model 
    train_loader = he.get_loader(name=exp_dict['dataset'], split='train', 
                                 datadir=os.path.dirname(savedir),
                                 exp_dict=exp_dict)
    model = he.get_model(name=exp_dict['model'], exp_dict=exp_dict)

    # 3. load checkpoint
    chk_dict = hw.get_checkpoint(savedir)

    # 4. Add main loop
    for epoch in tqdm.tqdm(range(chk_dict['epoch'], 10), 
                           desc="Running Experiment"):
        # 5. train for one epoch
        train_dict = model.train_on_loader(train_loader, epoch=epoch)

        # 6. get and save metrics
        score_dict = {'epoch':epoch, 'acc': train_dict['train_acc'], 
                      'loss':train_dict['train_loss']}
        chk_dict['score_list'] += [score_dict]

        images = model.vis_on_loader(train_loader)

    hw.save_checkpoint(savedir, score_list=chk_dict['score_list'], images=[images])
    print('Experiment done\n')

if __name__ == "__main__":
  # task 1 - submit example job
  """
  Run echo 35 and forward the logs to <savedir>/logs.txt
  """
  # command = 'echo 35'
  # savedir = '/home/xhdeng/shared/results/test_slurm/example_get_jobs'
  # job_id = submit_job(command, savedir)

  # # task 2 - get job info as dict
  # """
  # Get job info as dict and save it as json in the directory specified below
  # """
  # job_info = get_job(job_id)
  # hu.save_json('/home/xhdeng/shared/results/test_slurm/example/job_info.json', job_info)

  # task 3 - kill job
  # """
  # Kill job then gets its info as dict and save it as json in the directory specified below
  # it should say something like CANCELLED
  # """
  # kill_job(job_id)
  # job_info = get_job(job_id)
  # hu.save_json('/home/xhdeng/shared/results/test_slurm/example_kill_job/job_info_dead.json', job_info)
  
  # task 4 - get all jobs from an account as a list
  # """  
  # Get all jobs from an account as a list and save it in directory below
  # """
  # job_info_list = get_jobs(job_configs.USER_NAME)
  # hu.save_json('/home/xhdeng/shared/results/test_slurm/example_get_jobs/job_info_list.json', job_info_list)

  # # task 5 - run 10 jobs using threads
  # """
  # Use thee parallel threads from Haven and run these jobs in parallel
  # """
  # pr = hu.Parallel()

  # for i in range(1,20):
  #   command = 'echo %d' % i
  #   savedir = '/home/xhdeng/shared/results/test_slurm/example_%d' % i
    
  #   pr.add(submit_job, command, savedir)

  # pr.run()
  # pr.close()

  # # task 6 with menu - run mnist experiments on these 5 learning rates
  savedir_base = "/home/xhdeng/shared/results/test_slurm/task6"
  import argparse

  parser = argparse.ArgumentParser()

  parser.add_argument('-ei', '--exp_id')
  args = parser.parse_args()
  exp_list = []
  for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 'bug']:
      exp_list += [{'lr':lr, 'dataset':'mnist', 'model':'linear'}]
  
  if args.exp_id is None:
    # run jobs
    
    print("\nTotal Experiments:", len(exp_list))
    prompt = ("\nMenu:\n"
              "  0)'ipdb' run ipdb for an interactive session; or\n"
              "  1)'reset' to reset the experiments; or\n"
              "  2)'run' to run the remaining experiments and retry the failed ones; or\n"
              "  3)'status' to view the job status; or\n"
              "  4)'kill' to kill the jobs.\n"
              "Type option: "
              )
    
    option = input(prompt)
    if option == 'run':
      # only run if job has failed or never ran before
      pr = hu.Parallel()
      for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)

        # save exp folder
        save_exp_folder(exp_dict, savedir_base, False)

        # check job status
        job_id = get_job_id(exp_id, savedir_base)
        if job_id != -1:
          job_info = get_job(job_id)
          # job_info == 'Job not running'
          if job_info["JobState"] == "RUNNING" or job_info["JobState"] == "PENDING":
            # job is running, no need to submit again
            continue
        
        # copy code to savedir_base/code
        # workdir = os.path.join(savedir_base, exp_id, 'code')
        # currentdir = os.path.join(hu.subprocess_call("pwd"))
        # hu.copy_code(currentdir, workdir)
        
        command = 'python test_slurm.py -ei %s' % exp_id
        savedir = os.path.join(savedir_base, exp_id)
        pr.add(submit_job, command, savedir)

      pr.run()
      pr.close()

    elif option == 'reset':
      # ressset each experiment (delete the checkpoint and reset)
      pr = hu.Parallel()
      for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        # check for job status before reset and kill the job if submitted
        job_id = get_job_id(exp_id, savedir_base)
        if job_id != -1:
          job_info = get_job(job_id)
          if job_info["JobState"] == "RUNNING" or job_info["JobState"] == "PENDING":
            kill_job(job_info["JobState"])

        # reset folder
        save_exp_folder(exp_dict, savedir_base, True)

        # copy code to savedir_base/code
        workdir = os.path.join(savedir_base, exp_id, 'code')
        currentdir = os.path.join(hu.subprocess_call("pwd"))
        hu.copy_code(currentdir, workdir)

        command = 'python test_slurm.py -ei %s' % exp_id
        savedir = os.path.join(savedir_base, exp_id)
        pr.add(submit_job, command, savedir)

      pr.run()
      pr.close()

    elif option == 'status':
      # get job status of each exp
      job_list = []
      for i in range(1, len(exp_list) + 1):
        exp_id = hu.hash_dict(exp_list[i-1])
        savedir = os.path.join(savedir_base, exp_id)
        job_id = get_job_id(exp_id, savedir_base)
        if job_id != -1:
          job_info = get_job(job_id)
          output_string = ("Experiment %d/%d\n "
          "==================================================\n"
          "exp_id: %s\n"
          "job_id: %s\n"
          "job_state: %s\n"
          "savedir: %s\n"
          "exp_dict\n"
          "--------------------------------------------------\n"
          "%s\n") % (i, len(exp_list), exp_list[i-1], 
          job_info["JobId"], job_info["JobState"], savedir, job_info)

          print(output_string) 

    elif option == 'kill':
      # make sure all jobs for the exps are dead
      for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        job_id = get_job_id(exp_id, savedir_base)
        if job_id == -1:
          continue
        kill_job(job_id)

  else:
    for exp_dict in exp_list:
      exp_id = hu.hash_dict(exp_dict)
      savedir = '/home/xhdeng/shared/results/test_slurm/task6/%s' % exp_id
      if exp_id is not None and exp_id == args.exp_id:
        trainval(exp_dict, savedir, args={})
   
  

 