from setuptools import find_packages,setup
from typing import List
hyphen_e_dot = "-e ."

def get_requirements(file_path:str) ->List[str]:
    "this will return the lsit of requirements.txt"
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements= [req.replace("/n","") for req in requirements] ## using list comprehension
        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
    return requirements




setup(
    name = "machine_learning_project",
    version = '0.0.1',
    author = "Abhay Thakur",
    author_email= "rajputjiabhay3002@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)