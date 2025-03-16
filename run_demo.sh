#!/bin/bash
# Knowledge Graph QA System Demo Script

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display title
echo -e "${BLUE}=====================================================${NC}"
echo -e "${GREEN}Knowledge Graph QA System Demo - Based on Reinforcement Learning and LLM${NC}"
echo -e "${BLUE}=====================================================${NC}"

# Check environment
echo -e "${YELLOW}Checking environment settings...${NC}"
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Warning: OPENAI_API_KEY environment variable is not set${NC}"
    echo -e "${YELLOW}Would you like to set the API key? (y/n)${NC}"
    read answer
    if [ "$answer" = "y" ]; then
        echo -e "${YELLOW}Please enter your OpenAI API key:${NC}"
        read api_key
        export OPENAI_API_KEY=$api_key
        echo -e "${GREEN}API key has been set!${NC}"
    else
        echo -e "${RED}API key not set, program might not work correctly${NC}"
    fi
fi

# Check data directory
if [ ! -d "data" ]; then
    echo -e "${RED}Error: data directory 'data' does not exist${NC}"
    echo -e "${YELLOW}Please ensure you have downloaded the MetaQA dataset and placed kb.txt in the data directory${NC}"
    exit 1
fi

if [ ! -f "data/kb.txt" ]; then
    echo -e "${RED}Error: Knowledge graph file 'data/kb.txt' does not exist${NC}"
    echo -e "${YELLOW}Please ensure you have downloaded the MetaQA dataset and placed kb.txt in the data directory${NC}"
    exit 1
fi

# Create necessary directories
mkdir -p logs models

# Display options menu
echo -e "${GREEN}Please choose a demo mode:${NC}"
echo -e "${YELLOW}1) Answer a single question${NC}"
echo -e "${YELLOW}2) Run example questions set${NC}"
echo -e "${YELLOW}3) Interactive demo${NC}"
echo -e "${YELLOW}q) Quit${NC}"
read choice

case $choice in
    1)
        echo -e "${GREEN}Please enter a question:${NC}"
        read question
        echo -e "${GREEN}Please enter an entity (leave blank to attempt auto extraction):${NC}"
        read entity
        if [ -z "$entity" ]; then
            python3 run_demo.py --question "$question"
        else
            python3 run_demo.py --question "$question" --entity "$entity"
        fi
        ;;
    2)
        echo -e "${GREEN}Running example questions set...${NC}"
        python3 run_demo.py --run_examples
        ;;
    3)
        echo -e "${GREEN}Launching interactive demo...${NC}"
        python3 run_demo.py --interactive
        ;;
    q|Q)
        echo -e "${GREEN}Thank you for using this system! Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice, running default example${NC}"
        python3 run_demo.py
        ;;
esac

echo -e "${GREEN}Demo completed!${NC}"
echo -e "${BLUE}=====================================================${NC}"