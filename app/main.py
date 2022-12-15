import argparse
import os
from datetime import datetime

import fedstellar
from fedstellar.controller import Controller

argparser = argparse.ArgumentParser(description='Controller of Fedstellar framework', add_help=False)

argparser.add_argument('-f', '--federation', dest='federation', default="DFL",
                       help='Federation architecture: CFL, DFL, or SDFL (default: DFL)')
argparser.add_argument('-t', '--topology', dest='topology', default="fully",
                          help='Topology: fully, ring, random, or star (default: fully)')
argparser.add_argument('-w', '--webserver', dest='webserver', action='store_true', default=False,
                       help='Start webserver')
argparser.add_argument('-s', '--simulation', action='store_false', dest='simulation', help='Run simulation')
argparser.add_argument('-c', '--config', dest='config', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config'),
                       help='Config directory path')
argparser.add_argument('-l', '--logs', dest='logs', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'),
                          help='Logs directory path')
# Path to the file in same directory as this file
argparser.add_argument('-e', '--env', dest='env', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'), help='.env file path')
argparser.add_argument('-v', '--version', action='version',
                       version='%(prog)s ' + fedstellar.__version__, help="Show version")
argparser.add_argument('-tk', '--tracking', action='store_false', dest='tracking', help='Track simulation')
argparser.add_argument('-p', '--python', dest='python', default="/Users/enrique/miniforge3/envs/phd/bin/python", help='Path to python executable')
argparser.add_argument('-a', '--about', action='version',
                       version='Created by Enrique Tomás Martínez Beltrán',
                       help="Show author")
argparser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                       help='Show help')

args = argparser.parse_args()

'''
Code for deploying the controller 
'''
if __name__ == '__main__':
    # Change python path to the current environment (controller and participants)
    Controller(args).start()