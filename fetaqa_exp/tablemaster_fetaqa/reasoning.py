## Table Reasoning for Question Answering
import os
import sys
import json
import traceback
from tqdm import tqdm
import concurrent.futures
import numpy as np
import pandas as pd
import pandasql as ps
import re

sys.path.append('./')
from table_utils import format_table, column_to_index, cell_to_index
from azure_openai_api import get_openai_llm_response



CODE_MAX_RETRIES = 3



def reasoning_strategy_assessment(table_array, question):
    return False



fetaqa_examples = '''
Question: Who won the 1982 Illinois gubernatorial election, and how many votes was the margin?
Answer: Thompson prevailed in the 1982 Illinois gubernatorial election by a 5,074 vote margin.

Question: How did Michael and Mario Andretti do?
Answer: Michael Andretti finished with a run of 214.522 mph, faster than Mario.

Question: How many copies did "Pleasure" sell in 1998 alone, and how long was it the best selling album in Japan?
Answer: B'z The Best "Pleasure" sold more than 5 million copies in 1998 alone, making it a temporary best-selling album in Japanese music history, until being surpassed by Utada Hikaru's First Love in 1999.

Question: How many passengers can that plane hold?
Answer: The Tigerair Australia fleet consists of the following aircraft and Virgin Australia announced that the entire A320 fleet will be replaced with Boeing 737-800 aircraft.

Question: When and in what play did Platt appear at the Music Box Theatre?
Answer: In 2016 and 2017, Platt played in Dear Evan Hansen on Broadway at the Music Box Theatre.

Question: What are the download rates of EUTRAN?
Answer: EUTRAN has download rates of 299.6 Mbit/s and 150.8 Mbit/s.

Question: What roles did Melina Kanakaredes play in the television dramas; "Providence (1999–2002)", "CSI: NY (2004–2010)","Guiding Light"?
Answer: Melina Kanakaredes played in television dramas as Dr. Sydney Hansen in Providence (1999–2002) and as Detective Stella Bonasera in CSI: NY (2004–2010), and in the Guiding Light as Eleni Andros Cooper (1991–1995).

Question: What two teams did Austin Fyten play for during the 2015-16 season, and what league was the first team in?
Answer: In the 2015–16 season, Austin Fyten spent within the Bears and ECHL affiliate, the South Carolina Stingrays.

Question: What countries did the World U-17 Hockey Challenge attract after 2016?
Answer: The World U-17 Hockey Challenge attracted U-17 teams from Russia, Finland, Sweden, the United States, Canada, and the Czech Republic after 2016.

Question: Which club did Renato Hyshmeri play with in the 2014–15 Albanian Superliga season after leaving Partizani?
Answer: After playing with Partizani, Hyshmeri played with Tirana in the 2014–15 Albanian Superliga season.

Question: Which candidate won the 1990 Civic Forum leadership election and how many votes did they receive compared to their competitor?
Answer: Klaus won the 1990 Civic Forum leadership election, receiving 115 votes compared to 52 votes.

Question: Who were the top two candidates in the election, and how many votes did each receive?
Answer: John Nygren Republican defeated Joe Reinhard in the election (16,081–11,129).

Question: Where did Luchia Yishak place in the 3000m in the 1991 All-Africa Games? 
Answer: Luchia Yishak was the runner-up in the 3000 m at the 1991 All-Africa Games.

Question: Who were the sponsors for the away shirts of Colchester United FC?
Answer: Away shirt sponsorship has been provided by Ashby's (1999–2000), Ridley's (2000–2002), 188Trades.com (2005–2006), Smart Energy (2006–2009) and JobServe (2009–2010, 2012–).

Question: How much did the average attendance numbers for the Kansas City Comets vary?
Answer: The Kansas City Comets had a average high attendance of 15,786 in the 1983–1984 season and a low attendance of 7,103 in the 1990–1991 season.

Question: What roles did Blake Hood portray in the shows "90210" and "The Young and the Restless"?
Answer: Blake Hood played as Mark Driscoll on The 90210 and as Kyle Jenkins Abbott on The Young and the Restless.

Question: How well did Austin block perform during his two seasons with the Fairbanks ice dogs?
Answer: Block played with the Fairbanks Ice Dogs of the North American Hockey League for two seasons, where he had 100 points (42 goals, 58 assists) in 110 games, as he led the league in points (76) in 2008–09.

Question: Which roles did Bebe Neuwirth play in the 1986 Sweet Charity and which other role in the 1996 Chicago?
Answer: On stage, Bebe Neuwirth played the role of Nickie in Sweet Charity (1986) and Velma Kelly in Chicago (1996).

Question: What years did Ryan Kwanten appear in the films Flicka and as Jamie Ashen in Dead Silence?
Answer: Ryan Kwanten appeared in the films Flicka (2006) and as Jamie Ashen in Dead Silence (2007).

Question: What were Usian Bolts record finishes in the 200 meter, 400 meter and 4 × 100 m race?
Answer: In the boys' U-20 category, Usain Bolt from Jamaica records finishing the 200 meters in 20.43s, the 400 meters in 46.35s, and together with the Jamaican 4 × 100 m relay team in 39.43s.
'''


def free_form_qa(table_array, verbalized_table, question, examples=fetaqa_examples):
    QA_FORMATTING_PROMPT_TEMPLATE = """## Objective
You are given a table, a question, and a set of examples.
Your task is to answer the question based on the information provided in the table and the format of the examples.

## Table
{table}

## Verbalized Table
{verbalized_table}

## Question
{question}

## Instructions
1. Your answer must refer to the examples for formatting, style, text length, etc.
2. Construct your answer based on the data in the table while adhering to the examples' characteristics.
3. Your answer must be in a single sentence.
4. Your answer must be concise and avoid generating excessive text.
5. You must provide an answer even if the table does not contain the necessary information.

## Examples
{examples}

Now, answer the question based on the given table:
Question: {question}
Answer: """
    table_md = format_table(table_array, with_address=False)
    
    final_answer = get_openai_llm_response(QA_FORMATTING_PROMPT_TEMPLATE.format(table=table_md, verbalized_table=verbalized_table, question=question, examples=examples), json_output=False)
    return final_answer



# adaptive_program_aided_reasoning
def table_reasoning_for_qa(table_array, verbalized_table, question, task='qa', question_i=0):
    program_aided = reasoning_strategy_assessment(table_array, question)


    textual_reasoning_process = free_form_qa(table_array, verbalized_table, question)
    symbolic_reasoning_process = ''
    final_answer = textual_reasoning_process

    return final_answer, textual_reasoning_process, symbolic_reasoning_process



if __name__ == '__main__':
    with open('./test_tables/test2.json', 'r') as f:
        data = json.load(f)
    # table_array = data['table']
    table_array = [['Year', 'Nominated work', 'Result'], ['2007', 'Leona Lewis', 'Won'], ['2007', '"Bleeding Love"', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', '"Bleeding Love"', 'Won'], ['2008', '"Bleeding Love"', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', '"Bleeding Love"', 'Won'], ['2008', '"Spirit"', 'Nominated'], ['2008', 'Leona Lewis', 'Won'], ['2009', 'Leona Lewis', 'Nominated'], ['2009', 'Leona Lewis', 'Won'], ['2009', 'Leona Lewis', 'Won'], ['2009', '"Bleeding Love"', 'Won'], ['2009', 'Leona Lewis', 'Won'], ['2009', '"Bleeding Love"', 'Won'], ['2009', 'Leona Lewis', 'Won'], ['2009', 'Leona Lewis', 'Won']]
    question = data['question']
    example = 'Question: how many ships were launched in the year 1944?\nAnswer: 9'
    verbalized_table = 'The table provides a list of nominations and results for the artist Leona Lewis and her songs "Bleeding Love" and "Spirit" over the years 2007, 2008, and 2009. In 2007, Leona Lewis won for her work, and specifically for the song "Bleeding Love". Moving on to 2008, Leona Lewis continued her winning streak with multiple wins for her work and the song "Bleeding Love". Additionally, she was nominated for the song "Spirit" in the same year. In 2009, Leona Lewis was nominated for her work and won for both "Bleeding Love" and her other songs. Overall, Leona Lewis had a successful run with multiple wins and nominations for her music over the years.'
    final_answer, textual_reasoning_process, symbolic_reasoning_process = table_reasoning_for_qa(table_array, verbalized_table, question)
    print(final_answer)
    print(textual_reasoning_process)
    print(symbolic_reasoning_process)