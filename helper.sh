#!/bin/bash

# =============================================================================
# HELPER ACTIONS
# =============================================================================

NC=$(echo "\033[m")
BOLD=$(echo "\033[1;39m")
CMD=$(echo "\033[1;34m")
OPT=$(echo "\033[0;34m")

action_usage(){

    echo -e "   ____              _ ____  ____                                  "
    echo -e "  / ___|  __ _ _   _(_)  _ \/ ___|  Synthetic dataset generator    "
    echo -e "  \___ \ / _\` | | | | | | | \\___ \\  for Computer Vision tasks:  "
    echo -e "   ___) | (_| | |_| | | |_| |___) |   - detection;                 "
    echo -e "  |____/ \__, |\__,_|_|____/|____/    - localization;              "            
    echo -e "            |_|                       - segmentation;              "
    echo -e ""                                          
    echo -e "${BOLD}System Commands:${NC}"
    echo -e "   ${CMD}init${NC} initializers environment;"
    echo -e "   ${CMD}test${OPT} ...${NC} runs tests;"
    echo -e "      ${OPT}-m <MARK> ${NC}runs tests for mark;"
    echo -e "      ${OPT}-c ${NC}generates code coverage summary;"
    echo -e "      ${OPT}-r ${NC}generates code coverage report;"
    echo -e "   ${CMD}generate${OPT} -h${NC} generates synthetic dataset;"
    echo -e "   ${CMD}transform${OPT} -h${NC} transforms source to TFRecords;"
    echo -e "   ${CMD}docs${NC} generates documentation;"
    echo -e "   ${CMD}build${NC} generates distribution archives;"
    echo -e "   ${CMD}stage${NC} deploy SquiDS to Test Python Package Index;"  
}

action_init(){
    if [ -d .venv ];
        then
            rm -r .venv
    fi

    python3 -m venv .venv
    source .venv/bin/activate 

    if [[ -f dependencies.txt ]]
    then
        pip3 install -r dependencies.txt --no-cache
    else
        pip3 install -r requirements.txt --no-cache
    fi
}

action_test(){
    source .venv/bin/activate

    OPTS=()
    while getopts ":m:cr" opt; do
        case $opt in
            m)
                OPTS+=(-m $OPTARG) 
                ;;
            c)
                OPTS+=(--cov=squids) 
                ;;
            r)
                OPTS+=(--cov-report=xml:cov.xml) 
                ;;
            \?)
                echo -e "Invalid option: -$OPTARG"
                exit
                ;;
        esac
    done
    
    pytest --capture=no -p no:warnings ${OPTS[@]}
}

action_generate(){
    source .venv/bin/activate
    python main.py generate ${@}
} 

action_transform(){
    source .venv/bin/activate
    python main.py transform ${@}
}

action_mkdocs(){
    source .venv/bin/activate
    mkdocs serve
}

action_build(){
    source .venv/bin/activate
    python -m build
}

# =============================================================================
# HELPER COMMANDS SELECTOR
# =============================================================================
case $1 in
    init)
        action_init
    ;;
    test)
        action_test ${@:2}
    ;;
    generate)
        action_generate ${@:2}
    ;;
    transform)
        action_transform ${@:2}
    ;;
    mkdocs)
        action_mkdocs ${@:2}
    ;;
    *)
        action_usage
    ;;
esac  

exit 0