#!/bin/bash
name=$1
cur=$PWD
{
  cd /home/ros/kjx/HSPNav;
}
git add .  
git commit -m "${name}"  
git push origin  
cd "${cur}"
