import os
import sys
# sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import setup_env
from pinecone_text.sparse import BM25Encoder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from datetime import datetime
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr


class MemoryChat:
    def __init__(self, pc_index_name):
        self.system_prompt='''
           Actúa como un asistente de inmigracion del gobierno de España. Debes guiar al usuario a través del proceso de obtención de la autorización de residencia que mejor se adapte a sus necesidades.
           El usuario comenzará la interaccion enviando una imagen o documento. Debes identificar el código y nombre del documento. El código y nombre del documento suele aparecer arriba, el código empieza
           por EX y el nombre está dentro de un rectangulo negro.
           A continuación, debes preguntar en español, ingles, frances, arabe, rumano, chino y ucraniano el idioma del usuario. Cada opcion de idioma debe estar escrita en el propio idioma. 
           Cuando lo indique, el resto de la interaccion sera en ese idioma.
           A continuación, deberas generar la siguiente informacion sobre el documento:
            - Objetivo y funcion del documento: para que sirve y en que parte de todo el procedimiento migratorio es necesario
            - A quien esta dirigido
            - Ayuda para completarlo, en caso de que haya campos a rellenar por el usuario
            - Documentación adicional asociada al documento
            - Siguientes pasos dentro del procedimiento migratorio
            '''
        self.pc_index, self.embeddings, self.encoding, self.cohere, self.model = setup_env(pc_index_name)

    
    def cohere_rerank_texts(self, texts, query, co_client, rerank_model='rerank-v3.5', top_n=None, threshold_score=0.3, debug=False):
        """
        Re-ranks a list of texts based on their similarity to a given query using the Cohere re-ranking API.

        Args:
            texts (list of str): A list of texts to be re-ranked.
            query (str): The query used to rank the texts by relevance.
            co_client (CohereClient): An instance of the Cohere client for accessing the rerank method.

        Returns:
            list of tuples: A list of tuples where each tuple contains:
                - original_index (int): The original position of the text in the input list.
                - new_rank (int): The new ranking based on relevance to the query.
                - document (dict): The document object, containing the text.
                - relevance_score (float): The relevance score assigned by the re-ranking model.
        """
        documents = [{'text': doc} for doc in texts]
        ranked_results = co_client.rerank(
            query=query,
            documents=documents,
            top_n=top_n,  # Rank all documents
            model=rerank_model,  # Specify the re-ranking model,
        )
        tmp_len = len(ranked_results.results)
        ranked_results = [response for response in ranked_results.results if response.relevance_score >= threshold_score]
        if tmp_len != len(ranked_results) and debug:
            print(f'Se han eliminado {tmp_len - len(ranked_results)} chunks que no superaron el tresholds')
            
        # Create a list of tuples with original index, new rank, document, and relevance score
        ranked_results = [(idx, rerank.index + 1, documents[rerank.index]['text'], rerank.relevance_score) for idx, rerank in enumerate(ranked_results, 1)]
        
        return ranked_results
    
        
    def hybrid_scale(self, dense, sparse, alpha: float):
        if alpha < 0 or alpha > 1:
            raise ValueError('Alpha must be between 0 and 1')
        hsparse = {
            'indices': sparse['indices'],
            'values':  [v * (1 - alpha) for v in sparse['values']]
        }
        hdense = [v * alpha for v in dense]
        
        return hdense, hsparse
        
        
    def hybrid_query(self,index, question, embedding_model, fitted_bm25, top_k, alpha, filter={}):
        sparse_vec = fitted_bm25.encode_documents([question])[0]
        dense_vec = embedding_model.embed_documents([question])[0]
        dense_vec, sparse_vec = self.hybrid_scale(dense_vec, sparse_vec, alpha)
        result = index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            filter=filter,
            top_k=top_k,
            include_metadata=True
        )
        
        return result

    
    def get_context(self, question):
        bm25 = BM25Encoder().default()
        bm25.load('bm25_values.json')
        print(f"{datetime.now()}: Hybrid query...")
        results = self.hybrid_query(index=self.pc_index,question=question,embedding_model=self.embeddings,fitted_bm25=bm25,top_k=10,alpha=0.7)
    
        results_str = []
        for match in results['matches']:
            results_str.append(match['metadata']['text']) 
        results_str = '\n\n'.join(results_str)
        
        print(f"{datetime.now()}: Reranking...")
        reranked = self.cohere_rerank_texts(
            texts=results_str,
            query=question,
            co_client=self.cohere,
            rerank_model=os.getenv("COHERE_RERANK_MODEL"),
            top_n=5,
            threshold_score=0.3
        )
        return '\n\n'.join([f'Ranking {i}:\n{text[2]}' for i, text in enumerate(reranked, 1)][::-1])
         
            
    def call_model(self,state):
        response = self.model.invoke(state["messages"])
        return {"messages": response}


    def retrieval_node(self, state):
        if isinstance(state["messages"][-1], AIMessage):
            summary = state["messages"].pop(-1)
            state["messages"].insert(1,summary)
        questions = []
        if isinstance(state["messages"][1], AIMessage):
            questions.append(state["messages"][1].content)
        for x in state["messages"]:
            if isinstance(x, HumanMessage):
                questions.append(x.content)
                
        context = self.get_context('\n'.join(questions))
        question_with_context = f"""
            <pregunta>
                {state['messages'][-1].content}
            </pregunta>
            <contexto>
                Aquí tienes el contenido complementario que has de usar para responder la pregunta:
                {context}
            <\contexto>"""
        messages = state["messages"][:-1] + [HumanMessage(content=question_with_context)] + [RemoveMessage(id = state["messages"][-1].id)]
        
        return {"messages": messages}
        

    def trimmer_node(self, state):
        if len(state["messages"]) > 8:
            print(f"{datetime.now()}: Summarizing history...")
            system_message = state["messages"][0]
            last_interaction = state["messages"][-3:]
            message_history = state["messages"][1:-3]
            summary = self.model.invoke(message_history+[HumanMessage(content="Resume todos estos mensajes. Genera unicamente un listado con las ideas principales de la conversacion: Ejemplo: el usuario solicita informacion sobre sus jefes, el modelo indica que su jefe es Angel Fombellida, cuyo puesto es CEO")])
            print(f"{datetime.now()}: Summarized")
            delete_messages = [RemoveMessage(id=m.id) for m in message_history]
            trimmed_messages = delete_messages + [system_message, summary] + last_interaction
            return {"messages": trimmed_messages}
        return {"messages": state["messages"]}


    def setup_workflow(self):
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("trimmer", self.trimmer_node)
        workflow.add_node("retrieval", self.retrieval_node)
        workflow.add_node("model", self.call_model)
        workflow.add_edge(START, "trimmer")
        workflow.add_edge("trimmer", "retrieval")
        workflow.add_edge("retrieval", "model")
        
        app = workflow.compile(checkpointer = MemorySaver())

        return app        

    # def get_language(self, question):
    #     question = question.lower()
    #     if "español" in question:
    #         language="es"
    #     elif "english" in question:
    #         language="en"
    #     elif "français" in question:
    #         language="fr"        
    #     elif "العربية" in question:
    #         language="ar"        
    #     elif "română" in question:
    #         language="ro"        
    #     elif "中文" in question:
    #         language="zh"
    #     elif "українська" in question:
    #         language="uk"
            
    #     return language


    # def speech_to_text(self, language):
    #     r = sr.Recognizer() 
    #     try:
    #         with sr.Microphone() as source2:
    #             print("Listening...")
    #             r.adjust_for_ambient_noise(source2, duration=0.2)
    #             audio2 = r.listen(source2)
    #             text = r.recognize_google(audio2, language=language)
        
    #             return text
            
    #     except sr.RequestError as e:
    #         print('Could not request results; {0}'.format(e))
    #     except sr.UnknownValueError:
    #         print('unknown error occurred')


    # def text_to_speech(self,text, language):
    #     try:
    #         myobj = gTTS(text=text, lang=language,slow=False)
    #         myobj.save("audio.mp3")
    #         playsound("audio.mp3")
    #     finally:
    #         os.remove("audio.mp3")
        

    # def get_question(self, mode, language=None):
    #     if mode=="text":
    #         question = input("USER: ")
    #     else:
    #         question = self.speech_to_text(language)
    #     return question

    
    def chat(self):
        app = self.setup_workflow()
        
        include_system_prompt_flag = True
        question=input("PREGUNTA: ")
        print(f"{datetime.now()}: START CHAT")
        while question!="salir":
            if include_system_prompt_flag:
                messages = [SystemMessage(content = self.system_prompt), HumanMessage(content=question)]            
                include_system_prompt_flag = False
            else:
                messages = [HumanMessage(content=question)]       
                     
            result = app.invoke(
                {"messages": messages},
                config={"configurable": {"thread_id": "1"}}
            )
            
            print(f"PREGUNTA: {question}")
            print(f"RESPUESTA: {result['messages'][-1].content}")
            print("----------------------------------")
            question=input("PREGUNTA: ")
        print(f"{datetime.now()}: END CHAT")
