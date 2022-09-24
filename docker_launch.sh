#!/bin/bash

if [[ -f webui-user.sh ]]
then
    source ./webui-user.sh
fi

# Pretty print
delimiter="################################################################"

# printf "Create and activate python venv"
# if [[ ! -d "${venv_dir}" ]]
# then
#     "${python_cmd}" -m venv "${venv_dir}"
#     first_launch=1
# fi
# # shellcheck source=/dev/null
# if [[ -f "${venv_dir}"/bin/activate ]]
# then
#     source "${venv_dir}"/bin/activate
# else
#     printf "\n%s\n" "${delimiter}"
#     printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"
#     printf "\n%s\n" "${delimiter}"
#     exit 1
# fi


printf "\n%s\n" "${delimiter}"
printf "Launching launch.py..."
printf "\n%s\n" "${delimiter}"
"${python_cmd}" launch.py --no-half
