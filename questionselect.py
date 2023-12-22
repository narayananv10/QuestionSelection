import pandas as pd
import numpy as np
import json
import firebase_admin
import pandas as pd
import random
import requests
import ast
import fire
import pymongo
import pickle
import itertools
from firebase_admin import credentials, firestore, storage
from random import choice
from string import ascii_uppercase
from adaptive_learning.scheduler import DashScheduler
from urllib.parse import quote_plus

cred = credentials.Certificate("machine-learning-database.json")
app = firebase_admin.initialize_app(cred,{'storageBucket': 'machine-learning-databas-9d23e.appspot.com'})
firestore_client = firestore.client()

       
class QuestionSelect:
    """ 
    This class is used to select the suitable questions based on the student's study histories and adaptively according
    to the changing educational context.
    
    Attributes:
    -----------
    course_name: str
        The name of the course.
    dash_params : obj
        The parameters to be passed to the DASH API for computing the recall probability.
    concept_file : obj
        The Concept File for the course under consideration. """

    
    def __init__(self, course_name, dash_path, concept_path):
        """ 
        Initializes the class with the required attributes.
        
        Attributes:
        -----------
        course_name: Name of the course (ex: SIADS 542)
        
        dash_path: Path for the dash_params.pkl file
        
        concept_path: Path for the concept network generated for the course, also in the form of a .pkl file"""
        
        self.course_name = course_name
        self.dash_params = self.load_pickle(dash_path)
        self.concept_file = self.load_pickle(concept_path)
    
    def load_pickle(self, pickle_file_path):
        """
        Load saved data from a .pickle file.
        
        Parameters:
        -----------
        pickle_file_path : str
            Path of the .pickle file.
            
        Returns:
        -----------
        dict
            Returns the loaded data from the .pickle file."""
        
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    
    def GetQuestions(self):
        """
        Fetches all the questions related to the course from the firestore database.

        Returns:
        --------
        obj
            Returns the dataframe of all the questions from the firestore database. """
        
        questions_dict = {}
        course_list = []
        course = self.course_name

        ref = firestore_client.collection("Courses").document(course).collection("Lectures")
        for lec in ref.get():
            ref = firestore_client.collection("Courses").document(course).collection("Lectures").document(lec.id).collection("Moments")
            for moment in ref.get():
                ref = firestore_client.collection("Courses").document(course).collection("Lectures").document(lec.id).collection("Moments").document(moment.id).collection("Multiple Choice Questions")
                for q in ref.get():
                    questions_dict[q.id] = q.to_dict()
                    course_list.append(course)

        question_statements = []
        A_list = []
        B_list = []
        C_list = []
        D_list = []
        week_list = []
        correct_list = []
        moment_list = []
        question_topics = []
        for q in questions_dict.values():
            question_statements.append(q["Question"])
            A_list.append(q["A"])
            B_list.append(q["B"])
            C_list.append(q["C"])
            D_list.append(q["D"])
            correct_list.append(q["Correct answer"])
            moment_list.append(q["Moment"])
            question_topics.append(q["Topics"])

        quest_df = pd.DataFrame({
        "Question ID": questions_dict.keys(),
        "Question": question_statements,
        "A": A_list,
        "B": B_list,
        "C": C_list,
        "D": D_list,
        "Correct": correct_list,
        "Moment" : moment_list,
        "Topics" : question_topics})

        quest_df['Selected'] = False
        quest_df['Topics'] = quest_df['Topics'].apply(lambda topics: [topic.lower() for topic in topics])

        return quest_df

    def explode_list(self, df, column_to_explode):
        """
        A helper function used in preprocessing to explode a dataframe on a given column.
        
        Parameters:
        -----------
        df : obj
            The dataframe to be exploded.
        
        column_to_explode : str
            The column on the dataframe to be exploded.
            
        Returns:
        -----------
        obj
            The exploded dataframe."""
        
        df = df.reset_index(drop=True)
        s = df[column_to_explode]
        i = np.arange(len(s)).repeat(s.str.len())
        return df.iloc[i].assign(**{column_to_explode: np.concatenate(s)})


    def get_dash_memory(self, dash_params,concepts, progress):
        """
        Helper function used in GetFrequency to Get the recall probabilities from Dash API.
        
        Parameters:
        ----------
        dash_params : obj
            The parameters to be passed to the DASH API for computing the recall probability.
        concepts : obj
            The Concept file for the course under consideration.
        progress : dict
            The progress of the student.
            
        Returns:
        --------
        dict
            A dictionary having the recall probabilities for each concept."""
        
        if progress is None: # if progress is not provided...
            # then, provide a default value for progress (has to be in this form)
            progress = {'progress': [(5, 0, '08/08/2023'), (16, 1, '08/09/2023'), (5, 0, '08/10/2023'), (9, 0, '08/11/2023'), (4, 1, '08/12/2023'),
                             (0, 1, '08/13/2023'), (10, 1, '08/14/2023'), (11, 1, '08/15/2023'), (19, 0, '08/16/2023'), (10, 1, '08/17/2023'),
                             (18, 0, '08/18/2023'), (3, 1, '08/19/2023'), (1, 1, '08/20/2023'), (5, 1, '08/21/2023'), (5, 0, '08/22/2023'),
                             (0, 1, '08/23/2023'), (17, 1, '08/24/2023'), (5, 0, '08/25/2023'), (2, 0, '08/26/2023'), (16, 1, '08/27/2023'),
                             (1, 0, '08/28/2023'), (12, 0, '08/29/2023'), (20, 1, '08/30/2023'), (7, 0, '08/31/2023'), (14, 1, '09/01/2023'),
                             (20, 1, '09/02/2023'), (8, 1, '09/03/2023'), (8, 1, '09/04/2023'), (4, 0, '09/05/2023'), (7, 1, '09/06/2023')]
                        }
        scheduler = DashScheduler(concepts, dash_params)
        return scheduler.get_memory(progress['progress'])    

    def GetFrequency(self, df, student_uniqname):
        """
        Calculate the required frequency of each question based on the student responses 
        and concept recall probabilities.
        
        Parameters:
        ----------
        df : obj
            The response data for the student.
        student_uniqname : str
            The unique name/ID of the student.

        Returns:
        --------
        dict
            The required frequency for each topic."""
        
        rec_prob={}
        
        #Creating concept name to concept id dictionary
        id2concept = {k: c for k, c in enumerate(self.concept_file.nodes())}    
        id2concept = {k: v.strip() for k, v in id2concept.items()}
        concept2id = {v.strip(): k for k, v in id2concept.items()}
        
        output = None
        if df is None or df.empty:
            # no df provided, so get_dash_memory uses its default progress
            output = self.get_dash_memory(self.dash_params,self.concept_file,None)
            
        else:
            # Only process df if it's not None,
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%m/%d/%Y')
            df['correct/incorrect'] = (df['response'] == df['correct_answer']).astype(int)
            df = df[['topics_covered', 'correct/incorrect', 'timestamp']]

            #Format student history in suitable format for dash api
            df_formatted = self.explode_list(df, 'topics_covered')
            df_formatted.columns = ['concept', 'correct/incorrect', 'timestamp']
            df_formatted.reset_index(drop=True, inplace=True)
            df_formatted['concept'] = df_formatted['concept'].map(concept2id)
            df_formatted = df_formatted[df_formatted['concept'].apply(lambda x: str(x).isdigit())]
            df_formatted['concept'] = df_formatted['concept'].astype(int)
            tuple_list = [tuple(x) for x in df_formatted.values]
            student_hist = {'progress': tuple_list}

            # Get recall probabilities from the api
            output = self.get_dash_memory(self.dash_params,self.concept_file,student_hist)
            
            
#         #student history data from the database (code for connecting an online database must be added here)
#         df = df[df['student_id']==student_uniqname]          #filter by student
#         df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%m/%d/%Y')
#         df['correct/incorrect'] = (df['response'] == df['correct_answer']).astype(int)
#         df = df[['topics_covered', 'correct/incorrect', 'timestamp']]
        
#         #Format student history in suitable format for dash api
#         df_formatted = self.explode_list(df, 'topics_covered')
#         df_formatted.columns = ['concept', 'correct/incorrect', 'timestamp']
#         df_formatted.reset_index(drop=True, inplace=True)
#         df_formatted['concept'] = df_formatted['concept'].map(concept2id)
#         df_formatted = df_formatted[df_formatted['concept'].apply(lambda x: str(x).isdigit())]
#         df_formatted['concept'] = df_formatted['concept'].astype(int)
#         tuple_list = [tuple(x) for x in df_formatted.values]
#         student_hist = {'progress': tuple_list}
        
#         #Get recall probabilities from the api
#         output = self.get_dash_memory(self.dash_params, self.concept_file, student_hist['progress'])


        for i in output:
            k = int(i[1])
            v = float(i[0])
            rec_prob[k] = v
            
        #Formatting the recall probabilites to frequency in the correct format
        recall_prob_dict = {id2concept.get(k, k): v for k, v in rec_prob.items()}
        recall_prob_dict = {k.strip(): v for k, v in recall_prob_dict.items()}
        recall_prob_dict = dict(sorted(recall_prob_dict.items(), key=lambda item: item[1]))
        first_25_dict = dict(itertools.islice(recall_prob_dict.items(), 25))
        frequencies = {k: round(1/(v**0.5)) for k,v in first_25_dict.items()} #formula for converting the probabilities to frequency
        return frequencies

    def QuestionSelect(self, frequencies, quest_df, max_questions):
        """
        Select questions based on the calculated frequencies from GetFrequency.
        
        Parameters:
        ----------
        frequencies : dict
            The required frequency for each topic.
        quest_df : obj
            The dataframe having all the questions and their metadata information.
        max_questions : int
            The maximum number of questions to be selected for the student.
            
        Returns:
        --------
        obj
            The selected questions in the dataframe format."""
        
        # Start with all topic frequencies being the target ones
        unsatisfied_freqs = {k: v for k, v in frequencies.items() if v != 0}
        selected_questions = []

        # Greedy selection of questions
        while unsatisfied_freqs and len(selected_questions) < max_questions:

            # Calculate the score of each question by conditionally filtering unselected questions, 
            # and only if they cover a topic with unsatisfied frequency remaining
            question_scores = {q: sum(unsatisfied_freqs[topic] for topic in topics 
                                      if topic in unsatisfied_freqs and unsatisfied_freqs[topic] > 0)
                               for q, topics in quest_df[quest_df.Selected == False].set_index('Question').Topics.items() 
                               if any(topic in unsatisfied_freqs and unsatisfied_freqs[topic] > 0 for topic in topics)}

            # If no question can satisfy the remaining unsatisfied frequencies, then break the loop
            if not question_scores:
                print("No more questions can satisfy the remaining topic frequencies.")
                break

            # Select the question with the highest score
            selected_q = max(question_scores, key=question_scores.get)
            selected_questions.append(selected_q)

            # Update the 'Selected' flag for the chosen question
            quest_df.loc[quest_df.Question == selected_q, 'Selected'] = True

            # Update the unsatisfied frequencies
            for topic_list in quest_df.set_index('Question').loc[selected_q].Topics:  # Assume here each topic_list is a list
                for topic in topic_list:  # Iterate over the items in each topic_list
                    if topic in unsatisfied_freqs:
                        unsatisfied_freqs[topic] -= 1
                        if unsatisfied_freqs[topic] == 0:
                            unsatisfied_freqs.pop(topic)

        selected_questions_df = pd.DataFrame(selected_questions, columns=['Question'])
        final_df = selected_questions_df.merge(quest_df, on='Question', how='left')
        return final_df
