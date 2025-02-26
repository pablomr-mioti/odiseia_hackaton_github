import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pinecone_text.sparse import BM25Encoder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from datetime import datetime
from langgraph.graph import StateGraph, MessagesState, START, END
from src.d01_data.data import *
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.documents import Document
from typing import List, Tuple
from langgraph.checkpoint.memory import MemorySaver
import re
from datetime import datetime
from deep_translator import GoogleTranslator


# class AgentWithTools(ABC):
class RagAgent:
    def __init__(self, pc_index_name, language="es", log_file=None):
        self.bm25 = BM25Encoder().default()
        self.bm25.load(os.path.dirname(os.path.abspath(__file__)) + '/../../data/04_model_output/bm25_values.json')
        self.language = language
        self.input_translator = GoogleTranslator(source=self.language, target="es")
        self.output_translator = GoogleTranslator(source="es", target=self.language)
        self.system_prompt=f'''
        Actúa como un asistente de inmigracion del gobierno de España. NUNCA hables del usuario en tercera persona
        Debes guiar al usuario a través del proceso de obtención de la autorización de residencia y trabajo por circustancias excepcionales.
        El usuario comenzara la interaccion indicando su pais de origen. Si no lo hace, debes preguntarlo. La conversacion no puede avanzar hasta que no conzcas el pais de origen.
        A continuacion, pregunta si conoce el tipo de autorizacion que necesita solicitar. En caso afirmativo, debera indicarlo. En caso negativo, preguntale por el motivo de emigrar a españa o su situacion administrativa para asignar el tipo de autorizacion que mejor se adapte a su situacion.
        '''
        self.pc_index, self.embeddings, self.encoding, self.cohere, self.model = setup_env(pc_index_name)     
        self.country=None
        self.auth_type=None
        self.log_file = log_file
        self.auth_names = {
            "HI 35": "Autorización de residencia temporal por circunstancias excepcionales. Arraigo laboral",
            "HI 36": "Autorización de residencia temporal por circunstancias excepcionales. Arraigo social",
            "HI 37": "Autorización de residencia temporal por circunstancias excepcionales. Arraigo familiar",
            "HI 38": "Autorización de residencia temporal por circunstancias excepcionales por razones de protección internacional",
            "HI 39": "Autorización de residencia temporal por circunstancias excepcionales por razones humanitarias",
            "HI 40": "Autorización de residencia temporal por circunstancias excepcionales por colaboración con autoridades policiales, fiscales, judiciales o seguridad nacional",
            "HI 41": "Autorización de residencia temporal por circunstancias excepcionales por colaboración con autoridades administrativas o interés público y colaboración con la administración laboral",
            "HI 42": "Autorización de residencia temporal y trabajo por circunstancias excepcionales de mujeres extranjeras víctimas de violencia de género o de violencia sexual",
            "HI 43": "Autorización de residencia temporal y trabajo por circunstancias excepcionales por colaboración con autoridades administrativas no policiales, contra redes organizadas",
            "HI 44": "Autorización de residencia temporal y trabajo por circunstancias excepcionales por colaboración con autoridades policiales, fiscales o judiciales, contra redes organizadas",
            "HI 45": "Autorización de residencia temporal y trabajo por circunstancias excepcionales de extranjeros víctimas de trata de seres humanos",
            "HI 108": "Autorización de residencia temporal por circunstancias excepcionales. Arraigo para la formación",
        }


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
        
        
    def hybrid_query(self,index, question, embedding_model, top_k, alpha, filter):
        sparse_vec = self.bm25.encode_documents([question])[0]
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


    def get_context(self, question, filter={}):
        results = self.hybrid_query(index= setup_db("hackaton"),question=question,embedding_model=setup_embeddings(),top_k=20,alpha=0.6, filter=filter)

        results_str = []
        for match in results['matches']:
            results_str.append(match['metadata']['text']) 
        
        reranked = self.cohere_rerank_texts(
            texts=results_str,
            query=question,
            co_client=setup_cohere(),
            rerank_model='rerank-v3.5',
            top_n=5,
            threshold_score=0.4
        )
        return '\n\n'.join([f'Ranking {i}:\n{text[2]}' for i, text in enumerate(reranked, 1)][::-1])


    def get_country(self, state):
        country_response = self.model.invoke(f"Devuelve el pais de origen del usuario indicado en su prompt. Si no puedes, devuelve la cadena 'desconocido'. Prompt: {state['messages'][-1].content}")
        if country_response.content.lower() != 'desconocido':
            self.country = country_response.content        
            response = AIMessage(content = f"Gracias por indicar que tu país de origen, {self.country}. ¿Conoces el tipo de autorización de residencia y trabajo por circunstancias excepcionales que necesitas solicitar? Si no, por favor, indícame el motivo de tu emigración a España o tu situación administrativa actual para poder ayudarte")       
        else:
            response = AIMessage(content = f"Por favor, indica tu país de origen para continuar")   
        return {"messages": response}      


    def get_auth_types_info(self):
        text = ''
        path = os.path.dirname(os.path.abspath(__file__)) + '/../../data/02_intermediate/orientacion/'
        for doc in os.listdir(path):
            json_path = f'{path}/{doc}'
            doc_info = get_orientacion_docs(json_path, keys = ["Tipo de autorización", "Normativa básica", "Requisitos"])[1]
            text += "\n" + doc_info.replace("content: \n", "")
            
        return text


    def assign_auth_type(self, state):
        self.log_print(f"{datetime.now()}: assign auth type")
        text = self.get_auth_types_info()
        auth_type_response = self.model.invoke(f"""
                Actúa como un asistente de inmigracion del gobierno de España.
                Debes guiar al usuario a través del proceso de obtención de la autorización de residencia y trabajo por circustancias excepcionales. NUNCA hables del usuario en tercera persona
                Esto es un prompt del usuario: {state['messages'][-1].content}. Analizalo y usalo para identificar el tipo de autorizacion que necesita.
                Si el usuario indica el tipo de autorizacion que quiere solicitar, responder con el codigo y nombre completo, a quien va dirigido y un resumen de sus requisitos minimos. 
                Si el usuario indica el motivo por el que emigra a España, debes analizar por completo todos los tipos de autorizacion y responder con la que mejor se adapte a la situacion del usuario
                Pais de origen: {self.country}
                Tipos de autorizacion:{text}.
                Debes responder con el codigo y nombre, a quien va dirigido y un resumen de sus requisitos minimos.
                Si ningun tipo de autorizacion se adapta a la situacion del usuario, explica el motivo y indica el tipo de autorizacion mas parecido.""")
        try:
            self.auth_type = re.findall("HI \d+", auth_type_response.content)[0]
        except:
            self.auth_type = None
        return {"messages": auth_type_response}

    
    def get_historial(self, state):
        historial = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                historial.append(f"USER - {msg.content}")
            elif isinstance(msg, AIMessage):
                historial.append(f"LLM - {msg.content}")
        historial = '\n'.join(historial)      
        
        return historial  


    def search_auth_types(self, state):
        self.log_print(f"{datetime.now()}: search auth types")
        context = self.get_auth_types_info()
        response = self.model.invoke(f"""
                          <pregunta>
                            {state['messages'][-1]}
                          </pregunta>
                          <contexto>
                            {context}
                          </contexto>""")
        return {"messages":response}
   
            
    def router(self,state):
        
        if self.country:
            if self.auth_type:
                assign_auth_type = self.model.invoke(f"Si la frase del final de estas instrucciones hace referencia directa  un tipo de autorizacion distinto de {self.auth_type} - {self.auth_names[self.auth_type]}, devuelve la cadena SI. En caso contrario, devuelve la cadena NO. Una duda sobre el tipo de autorizacion actual no se considera una referencia a otro tipo de autorizacion. Los tipos de autorizacion son {self.auth_names}. Frase: {state['messages'][-1].content}").content
            else:
                assign_auth_type = "SI"
                
            if assign_auth_type == "SI":
               return "assign_auth_type"
            
            search_auth_types = self.model.invoke(f"Al final de estas instrucciones se encuentra el historial de mensajes. Si el ultimo mensaje del historial solicita directamente informacion sobre varios tipos de autorizacion, una comparativa o cualquier solicitud que implique varios tipos de autorizacion, devuelve la cadena SI. En caso contratio, devuelve la cadena NO. Ten en cuenta el historial para inferir el significado del ultimo mensaje. Los tipos de autorizacion son: {self.auth_names}. Historial: {self.get_historial(state)}").content
            
            if search_auth_types == "SI":
                return "search_auth_types"
            
            else:              
                return "retrieval"
               
        else:
            return "get_country"
        
   
    def retrieval(self, state: MessagesState) -> Tuple[str, List[Document]]:
        historial = self.get_historial(state)
        prompt = f"""
        Genera una string que funcione como query para obtener contexto relevante de una base de datos vectorial con informacion sobre leyes migratorias y proteccion internacional y informacion detallada sobre los distintos tipos de autorizacion que se pueden solicitar. Al final de estas instrucciones se encuentra el historial de mensajes entre un usuario y un modelo llm. Debes generar una query que permita responder a la ultima pregunta sin perder informacion importante de interacciones anteriores.
        Ejemplo:
        USER - estoy buscando trabajo para ampliar mi visado
        LLM - debes solicitar la Autorización de residencia temporal por circunstancias excepcionales. Arraigo laboral (HI 35)
        USER - Como lo solicito y que requisitos tiene?
        RESPUESTA: solicitud y requisitos Autorización de residencia temporal por circunstancias excepcionales. Arraigo laboral (HI 35)
        Historial: {historial}
        Devuelve SOLO el string solicitado
        """
        query = self.model.invoke(prompt).content
        filter = {"$or":[
            {"tema":{"$ne": "orientacion"}},
            {"fichero": self.auth_type}
        ]}
        context = self.get_context(query, filter)
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


    def generic_question(self, state):
        question = f"""
            <user info>
                Pais de origen: {self.country}
            </user info>
            {state["messages"][-1].content}"""  
        response = self.model.invoke(question)
        return {"messages": response}
    
    
    def translate(self,state):
        if self.language == "es":
            return state
        
        translated_input = self.input_translator.translate(state["messages"][-1].content)
        messages = state["messages"][:-1] + [HumanMessage(content = translated_input)] + [RemoveMessage(id = state["messages"][-1].id)]

        return {"messages": messages}
    
    
    def setup_workflow(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("translate", self.translate)
        workflow.add_node("get_country", self.get_country)
        workflow.add_node("search_auth_types", self.search_auth_types)
        workflow.add_node("assign_auth_type", self.assign_auth_type)
        workflow.add_node("retrieval", self.retrieval)
        workflow.add_node("generic_question", self.generic_question)
                
        workflow.add_edge(START, "translate")        
        workflow.add_conditional_edges("translate", self.router, ["get_country","search_auth_types","assign_auth_type", "retrieval"])
        workflow.add_edge("get_country", END)
        workflow.add_edge("search_auth_types", END)
        workflow.add_edge("assign_auth_type", END)
        workflow.add_edge("retrieval", "generic_question")
        workflow.add_edge("generic_question", END)
        
        app = workflow.compile(checkpointer = MemorySaver())
        
        return app
      
      
    def log_print(self, text, translate=False):
        if translate and self.language != "es":
            text = self.output_translator.translate(text)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(text)
                f.write("\n")
        print(text)


    def chat(self):
        app = self.setup_workflow()
        include_system_prompt_flag = True
        self.log_print(f"{datetime.now()}: START CHAT")
        self.log_print("AI: Hola, soy un asistente de inmigración del gobierno de España. Estoy aquí para ayudarte a obtener la autorización de residencia y trabajo por circunstancias excepcionales. Para comenzar, ¿puedes decirme cuál es tu país de origen?")
        question=input("PREGUNTA: ")      
        self.log_print(f"{datetime.now()}: USER - {question}")
        
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
            
            self.log_print(f"{datetime.now()}: AI - {result['messages'][-1].content}", translate=True)
            self.log_print("----------------------------------")
            question=input("PREGUNTA: ")
            self.log_print(f"{datetime.now()}: USER - {question}")
            
        self.log_print(f"{datetime.now()}: END CHAT")