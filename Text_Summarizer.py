#import modules
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

#initializing the pre trained models
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

# input text
text = """
At a very broad level, Data Analytics involves making decisions or coming to conclusions based on data that has been systemically organized. A more formal definition would include stages of the Data Analytics process, which include Planning, Collecting, Cleansing, Organizing, and Interpreting/Communicating. A deeper dive into the process will reveal how each state is integral to Data Analytics that produces value.

The Planning Phase is where the Data Analytics project details would be articulated. This includes any specific questions that need to be answered and methodologies that will be used. This phase is essential as it maps the details of the subsequent steps. For example, during the Planning phase, analysts will decide what data sources to use for the project. Not only will data sources be determined, but time periods of analysis as well. In other words, from what time period will data be pulled for use in the analysis? Most importantly, careful planning will ensure that the Data Analytics project is providing value that justifies the costs involved.

The Collecting Phase involves the gathering of the data that will be used in the analysis. This may include data stored internally within an organization, obtained through primary survey research, or gathered from external sources. The data may be quantitative or qualitative. Quantitative data is expressed in numbers as measures or counts, while qualitative data expresses traits or characteristics. The difference between the two is quantitative data defines, while qualitative data describes (BusinessDictionary.com).

In the Cleansing Phase, the collected data is checked for quality and usefulness. This phase is important as it has implications on the analysis of the data and ultimately the decisions and conclusions made from it.  Data cleansing checks for relevance, corruption, duplication and correct formatting. When possible, data should be corrected or fixed. However, if this is not possible then it should be excluded from the analysis to avoid compromising the validity of the data analysis. After completing the data cleansing, it is important to run reports to confirm the necessary changes were successfully executed and the data is not presenting contradictory information.

The preceding phases are vital and allow the Organizing Phase to run more smoothly. This phase is where the core of Data Analytics takes place. The cleansed data is organized and manipulated to find answers to the questions articulated in the Planning Phase. There are several types of Data Analytics tools that can be used to manipulate the data. Tools can be as simple as spreadsheet software such as Microsoft Excel or more advanced statistical software packages including SPSS or SAS. R, a programming language and software environment, has become a popular option for data analysts.

The final phase is the Interpreting/Communicating Phase. The first step in this phase is to develop a “story” using the answers found in the Organizing Phase. The “story should aid in the development of actionable steps. Decisions and conclusions made in this phase should be supported by the learnings discovered in the prior phase.

An important part of this final phase is communicating these discoveries, and ultimately the conclusions and recommendations, to the stakeholders involved. In most situations, final decisions are not made by one person, but a group. Often, at least some members in the group will have a limited understanding of Data Analytics and the methodologies involved. Therefore, it will be important to communicate the learnings and recommendations clearly and in an uncomplicated manner.

Data visualization will be an important component as the results of any Data Analytics project is communicated. The Visual Teaching Alliance has reported that 65% of the population are visual learners, while 30% have been found to be auditory learners. Including data visualizations in the mix, along with verbal communication, will ensure at least most of your audience’s learning needs are met. By doing so, the audience will more likely provide valuable feedback, and hopefully buy in to your recommendations.

As noted, each phase of the Data Analytics process is critical and therefore cannot stand-alone. It is important a Data Analyst has a thorough understanding and a high level of involvement in each phase to execute a successful Data Analytics project.
"""

#pre-process the input text
preprocessed_text = text.strip().replace('\n','')
t5_input_text = 'summarize: ' + preprocessed_text

#length of the input text
len_of_input = len(t5_input_text.split())
print('\nLength of the input text : ',len_of_input)

tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512).to(device)

#get user input for length of the summary
min_len = int(input('Enter the minimum length of the summary you need(in words) : '))
max_len = int(input('Enter the maximum length of the summary you need(in words) :'))

# text summarizer
summary_ids = model.generate(tokenized_text, min_length=min_len, max_length=max_len)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)