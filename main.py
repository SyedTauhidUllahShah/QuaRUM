import os
import re
import json
import time
import argparse
from enum import Enum
from typing import Dict,List,Tuple,Optional,Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from difflib import SequenceMatcher
class CSLRelationshipType(Enum):
    IS_A="is a"
    IS_PART_OF="is part of"
    PERFORMS="performs"
    IMPLEMENTATION="implements"
    DEPENDS_ON="depends on"
    USES="uses"
    ASSOCIATION="associates with"
    OWNS="owns"
    MANAGES="manages"
    CREATES="creates"
    ACCESSES="accesses"
class Code:
    def __init__(self,id:str,name:str,definition:str="",is_recommendation:bool=False):
        self.id=id
        self.name=name
        self.definition=definition
        self.attributes:List[Dict[str,str]]=[]
        self.methods:List[Dict[str,str]]=[]
        self.stereotypes:List[str]=[]
        self.trace_sources:List[str]=[]
        self.evidence_chunks:List[str]=[]
        self.is_interface:bool=False
        self.is_abstract:bool=False
        self.is_enumeration:bool=False
        self.enum_values:List[str]=[]
        self.confidence:float=0.0
        self.evidence_locations:List[str]=[]
        self.outgoing_relationships:List["CodeRelationship"]=[]
        self.incoming_relationships:List["CodeRelationship"]=[]
        self.relevance_score:float=0.0
        self.extracted_text:str=""
        self.is_recommendation=is_recommendation
        self.notes:List[str]=[]
class CodeRelationship:
    def __init__(self,id:str,source_code_id:str,target_code_id:str,relationship_type:CSLRelationshipType,association_name:str="",confidence:float=0.7,multiplicity:Dict[str,str]=None):
        self.id=id
        self.source_code_id=source_code_id
        self.target_code_id=target_code_id
        self.relationship_type=relationship_type
        self.association_name=association_name
        self.confidence=confidence
        self.multiplicity=multiplicity if multiplicity else {"source":"1","target":"*"}
        self.evidence_chunks:List[str]=[]
        self.evidence_locations:List[str]=[]
        self.extracted_text:str=""
class CodeSystem:
    def __init__(self):
        self.codes:Dict[str,Code]={}
        self.relationships:List[CodeRelationship]=[]
        self.domain_description=""
        self.core_domain_entities:List[str]=[]
        self.domain_actors:List[str]=[]
        self.domain_attributes:Dict[str,List[str]]={}
        self.domain_operations:Dict[str,List[str]]={}
        self.enumerations:Dict[str,List[str]]={}
        self.domain_relationships={}
        self.raw_text=""
    def add_code(self,code:Code)->str:
        for eid,existing_code in self.codes.items():
            if existing_code.name.lower()==code.name.lower() or SequenceMatcher(None,existing_code.name.lower(),code.name.lower()).ratio()>0.9:
                existing_code.attributes=list(set(json.dumps(attr,sort_keys=True) for attr in existing_code.attributes+code.attributes))
                existing_code.attributes=[json.loads(attr) for attr in existing_code.attributes]
                existing_code.methods=list(set(json.dumps(meth,sort_keys=True) for meth in existing_code.methods+code.methods))
                existing_code.methods=[json.loads(meth) for meth in existing_code.methods]
                existing_code.stereotypes=list(set(existing_code.stereotypes+code.stereotypes))
                existing_code.evidence_chunks.extend(code.evidence_chunks)
                existing_code.evidence_locations.extend(code.evidence_locations)
                existing_code.confidence=max(existing_code.confidence,code.confidence)
                if code.is_interface:existing_code.is_interface=True
                if code.is_abstract:existing_code.is_abstract=True
                if code.is_enumeration:
                    existing_code.is_enumeration=True
                    existing_code.enum_values=list(set(existing_code.enum_values+code.enum_values))
                if code.definition and not existing_code.definition:
                    existing_code.definition=code.definition
                if code.extracted_text:existing_code.extracted_text=code.extracted_text
                return eid
        self.codes[code.id]=code
        return code.id
    def add_relationship(self,relationship:CodeRelationship)->bool:
        if relationship.source_code_id not in self.codes or relationship.target_code_id not in self.codes:
            return False
        source=self.codes[relationship.source_code_id]
        target=self.codes[relationship.target_code_id]
        if relationship.relationship_type==CSLRelationshipType.IMPLEMENTATION and not target.is_interface:
            return False
        for r in self.relationships:
            if r.source_code_id==relationship.source_code_id and r.target_code_id==relationship.target_code_id and r.relationship_type==relationship.relationship_type:
                if relationship.confidence>r.confidence:
                    r.confidence=relationship.confidence
                    r.association_name=relationship.association_name
                    r.multiplicity=relationship.multiplicity
                    r.evidence_chunks.extend(relationship.evidence_chunks)
                    r.evidence_locations.extend(relationship.evidence_locations)
                    r.extracted_text=relationship.extracted_text
                return False
        self.relationships.append(relationship)
        source.outgoing_relationships.append(relationship)
        target.incoming_relationships.append(relationship)
        return True
    def calculate_relevance_scores(self):
        for code_id,code in self.codes.items():
            relation_count=len(code.incoming_relationships)+len(code.outgoing_relationships)
            code.relevance_score=min(relation_count*0.2,2.0)
            if code.name.lower() in [c.lower() for c in self.core_domain_entities]:
                code.relevance_score+=1.0
            code.relevance_score+=code.confidence*0.5
            if code.is_interface or code.is_abstract:
                code.relevance_score+=0.3
class DomainModelingPipeline:
    def __init__(self,api_key:str,model_name:str="gpt-4.1-mini"):
        self.api_key=api_key
        self.model_name=model_name
        self.domain_name=""
        self.domain_description=""
        self.code_system=CodeSystem()
        self.llm=None
        self.embeddings=None
        self.vector_store=None
        self.retriever=None
        self.documents=[]
        self.banned_words=set(["system","component","generic","entity","item","object","module"])
        self.SYSTEM_PROMPT="""You are a qualitative data analysis-based expert in UML and domain-driven design.
Your task is to extract domain models from natural language requirements.
Focus on creating clean, accurate, and useful class diagrams that properly represent the domain.
Identify the most important classes, attributes, operations, and relationships.
Ensure all JSON outputs are valid, with proper syntax (e.g., commas, brackets).
Use UML conventions for attributes and methods with visibility (+, -, #) and proper types."""
    def setup(self,file_path:str,domain_description:str):
        self.domain_description=domain_description
        self.code_system.domain_description=domain_description
        self.domain_name=self._extract_domain_name(domain_description)
        self._setup_langchain_components()
        self._load_document(file_path)
        self._analyze_domain()
        print(f"Setup complete for domain: {self.domain_name}")
        print(f"Core domain entities: {', '.join(self.code_system.core_domain_entities[:7])}")
        print(f"Loaded {len(self.documents)} document chunks")
    def _extract_domain_name(self,description:str)->str:
        match=re.search(r"([A-Za-z\s]+)(?:\s+system|\s+domain)?",description)
        return match.group(1).strip() if match else "System"
    def _setup_langchain_components(self):
        self.llm=ChatOpenAI(model=self.model_name,openai_api_key=self.api_key,temperature=0)
        self.embeddings=OpenAIEmbeddings(openai_api_key=self.api_key)
    def _load_document(self,file_path:str):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")
        with open(file_path,"r",encoding="utf-8") as f:
            text=f.read()
        self.code_system.raw_text=text
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        texts=text_splitter.split_text(text)
        self.documents=[]
        for i,chunk in enumerate(texts):
            self.documents.append(Document(page_content=chunk,metadata={"source":file_path,"chunk_id":f"chunk_{i}"}))
        self.vector_store=FAISS.from_documents(self.documents,self.embeddings)
        self.retriever=self.vector_store.as_retriever(search_type="similarity",search_kwargs={"k":5})
    def _extract_json_from_response(self,response:str)->Optional[dict]:
        try:
            cleaned_response=re.sub(r"```(?:json)?\n|```","",response).strip()
            json_block=re.findall(r"\{[\s\S]*\}",cleaned_response)
            if not json_block:
                print("No JSON object found in response")
                return None
            json_str=max(json_block,key=len)
            json_str=re.sub(r",\s*([\]}])",r"\1",json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {str(e)}")
            return None
    def _analyze_domain(self):
        prompt=f"""Analyze this domain in depth:
{self.domain_description}
Extract the following information to create a precise domain model:
1. Core Domain Entities: The primary domain objects (5-15, singular nouns, domain-specific)
2. Key Domain Relationships: Relationships between core entities with precise types
3. Key Domain Attributes: For each core entity, typical attributes with types and visibility
4. Key Domain Operations: For each core entity, main operations with signatures and visibility
5. Main Actors: Primary actors interacting with the system
6. Enumerations: Any enumerations with their values
Return as structured JSON with keys: "core_entities", "relationships", "attributes", "operations", "actors", "enumerations".
For "relationships", use format: [{{ "source": "EntityA", "target": "EntityB", "relationship": "verb phrase", "multiplicity": {{ "source": "1|0..1|*", "target": "1|0..1|*" }} }}]
For "attributes", use: {{ "Entity": [{{ "name": "attrName", "type": "String", "visibility": "+|-|#" }}] }}
For "operations", use: {{ "Entity": [{{ "name": "opName", "signature": "(param: Type): ReturnType", "visibility": "+|-|#" }}] }}
For "enumerations", use: {{ "EnumName": ["VALUE1", "VALUE2"] }}
Ensure JSON is valid and focuses on business domain concepts."""
        for attempt in range(3):
            try:
                docs=self.retriever.invoke(self.domain_description)
                domain_context="\n\n".join([doc.page_content for doc in docs[:3]])
                full_prompt=f"{prompt}\n\nAdditional domain information:\n{domain_context}"
            except:
                full_prompt=prompt
            response=self._call_llm(full_prompt)
            result=self._extract_json_from_response(response)
            if result:
                self.code_system.core_domain_entities=result.get("core_entities",[])
                relationships=result.get("relationships",[])
                for rel in relationships:
                    source=rel.get("source","")
                    target=rel.get("target","")
                    relationship=rel.get("relationship","")
                    multiplicity=rel.get("multiplicity",{"source":"1","target":"*"})
                    if source and target and relationship:
                        if source not in self.code_system.domain_relationships:
                            self.code_system.domain_relationships[source]=[]
                        self.code_system.domain_relationships[source].append(f"{relationship} {target} ({multiplicity['source']}:{multiplicity['target']})")
                self.code_system.domain_attributes=result.get("attributes",{})
                self.code_system.domain_operations=result.get("operations",{})
                self.code_system.domain_actors=result.get("actors",[])
                self.code_system.enumerations=result.get("enumerations",{})
                return
            print(f"Domain analysis attempt {attempt+1} failed, retrying...")
        print("Warning: Could not extract domain analysis JSON")
    def _verify_text_presence(self,term:str)->bool:
        if not term:return False
        pattern=re.compile(r'\b'+re.escape(term)+r'(?:s)?\b',re.IGNORECASE)
        return bool(pattern.search(self.code_system.raw_text))
    def _call_llm(self,prompt:str,system_prompt:str=None)->str:
        if system_prompt is None:
            system_prompt=self.SYSTEM_PROMPT
        prompt=prompt.replace("{","{{").replace("}","}}")
        chat_prompt=ChatPromptTemplate.from_messages([("system",system_prompt),("human",prompt)])
        chain=chat_prompt | self.llm
        try:
            response=chain.invoke({})
            return response.content
        except Exception as e:
            print(f"Error calling LLM: {str(e)}")
            return ""
    def _get_evidence_from_query(self,query:str,k:int=3)->List[Document]:
        try:
            return self.retriever.invoke(query)[:k]
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    def run_pipeline(self)->Tuple[str,str]:
        print("\n1. Starting Open Coding phase - Extracting entities...")
        self.perform_open_coding()
        print("Display results")
        self.display_entities_after_open_coding()
        print("\n2. Starting Axial Coding phase - Identifying relationships...")
        self.perform_axial_coding()
        print("\n3. Starting Selective Coding phase - Refining model...")
        self.perform_selective_coding()
        print("\n4. Generating UML diagram...")
        plantuml_code=self.generate_plantuml()
        report=self.generate_traceability_report()
        print(f"\nPipeline completed. UML model contains {len(self.code_system.codes)} classes and {len(self.code_system.relationships)} relationships")
        return plantuml_code,report
    def perform_open_coding(self):
        for i,doc in enumerate(self.documents):
            chunk_text=doc.page_content
            if len(chunk_text)<50:continue
            print(f"Processing chunk {i+1}/{len(self.documents)}...")
            domain_context=""
            if self.code_system.core_domain_entities:
                concepts_str=", ".join(self.code_system.core_domain_entities[:7])
                domain_context=f"\nKey domain concepts: {concepts_str}"
            prompt=f"""
# Domain Entity Extraction for {self.domain_name}
Analyze this text segment and extract key domain entities for a UML class diagram:
{chunk_text}
## Extraction Guidelines:
1. Focus on entities and concepts from the text
2. For each entity identify:
   - NAME: The name as mentioned in the text
   - DEFINITION: Description based on the text
   - ATTRIBUTES: Properties mentioned or implied in the text
   - OPERATIONS: Methods/behaviors mentioned or implied
   - TYPE: Class, Interface, or Enumeration
   - EVIDENCE: Text that mentions this entity
   - EXTRACTED_TEXT: Complete sentence where entity appears
   - CONFIDENCE: Rate evidence strength (0.5-1.0)
3. Include actors and interfaces
4. Look for attributes and methods that make sense in context
{domain_context}
Return JSON:
{{
"entities": [
    {{
    "name": "EntityName",
    "definition": "Description from text",
    "type": "Class|Interface|Enumeration|Actor",
    "attributes": [{{ "name": "attrName", "type": "Type", "visibility": "+|-|#" }}],
    "operations": [{{ "name": "opName", "signature": "(param: Type): ReturnType", "visibility": "+|-|#" }}],
    "evidence": "Text quote that mentions this entity",
    "extracted_text": "Complete sentence where entity appears",
    "confidence": 0.8
    }}
]
}}
For enumerations, include "enumValues": ["VALUE1", "VALUE2"] if listed in text."""
            for attempt in range(2):
                response=self._call_llm(prompt)
                entities=self._process_extracted_entities(response,doc)
                if entities:break
                print(f"Open coding attempt {attempt+1} failed for chunk {i+1}, retrying...")
            else:
                if self.code_system.core_domain_entities:
                    for entity_name in self.code_system.core_domain_entities[:2]:
                        if not any(code.name.lower()==entity_name.lower() for code in self.code_system.codes.values()):
                            new_id=f"code_{len(self.code_system.codes)+1}"
                            code_obj=Code(id=new_id,name=entity_name,definition=f"Core entity in {self.domain_name} domain",is_recommendation=True)
                            if entity_name in self.code_system.domain_attributes:
                                code_obj.attributes=self.code_system.domain_attributes[entity_name]
                            if entity_name in self.code_system.domain_operations:
                                code_obj.methods=self.code_system.domain_operations[entity_name]
                            if entity_name in self.code_system.domain_actors:
                                code_obj.stereotypes=["actor"]
                            else:
                                code_obj.stereotypes=["entity"]
                            code_obj.confidence=0.6
                            code_obj.evidence_chunks.append(chunk_text[:100])
                            code_obj.evidence_locations.append(f"{doc.metadata.get('source')}:{doc.metadata.get('chunk_id')}")
                            code_obj.notes.append("Added from domain analysis as core entity")
                            self.code_system.add_code(code_obj)
        self._ensure_core_entities_present()
        self._filter_entities()
        print(f"Open Coding complete. Extracted {len(self.code_system.codes)} candidate entities.")
    def display_entities_after_open_coding(self):
        print("\n===== ENTITIES AFTER OPEN CODING =====")
        enums = []
        interfaces = []
        abstract_classes = []
        classes = []
        actors = []
        for code_id, code in self.code_system.codes.items():
            if code.is_enumeration:
                enums.append(code)
            elif code.is_interface:
                interfaces.append(code)
            elif code.is_abstract:
                abstract_classes.append(code)
            elif "actor" in code.stereotypes:
                actors.append(code)
            else:
                classes.append(code)
        print(f"Total Entities: {len(self.code_system.codes)}")
        print(f"- Classes: {len(classes)}")
        print(f"- Interfaces: {len(interfaces)}")
        print(f"- Abstract Classes: {len(abstract_classes)}")
        print(f"- Enumerations: {len(enums)}")
        print(f"- Actors: {len(actors)}")
        if classes:
            print("\n== CLASSES ==")
            for code in sorted(classes, key=lambda x: x.name):
                print(f"Class: {code.name} (Confidence: {code.confidence:.2f})")
                if code.definition:
                    print(f"  Definition: {code.definition[:100]}...")
                if code.attributes:
                    print("  Attributes:")
                    for attr in code.attributes:
                        print(f"    {attr['visibility']} {attr['name']}: {attr['type']}")
                if code.methods:
                    print("  Methods:")
                    for method in code.methods:
                        print(f"    {method['visibility']} {method['name']}{method['signature']}")
                print()
        if interfaces:
            print("\n== INTERFACES ==")
            for code in sorted(interfaces, key=lambda x: x.name):
                print(f"Interface: {code.name} (Confidence: {code.confidence:.2f})")
                if code.definition:
                    print(f"  Definition: {code.definition[:100]}...")
                if code.methods:
                    print("  Methods:")
                    for method in code.methods:
                        print(f"    {method['visibility']} {method['name']}{method['signature']}")
                print() 
        if abstract_classes:
            print("\n== ABSTRACT CLASSES ==")
            for code in sorted(abstract_classes, key=lambda x: x.name):
                print(f"Abstract Class: {code.name} (Confidence: {code.confidence:.2f})")
                if code.definition:
                    print(f"  Definition: {code.definition[:100]}...")
                if code.attributes:
                    print("  Attributes:")
                    for attr in code.attributes:
                        print(f"    {attr['visibility']} {attr['name']}: {attr['type']}")
                if code.methods:
                    print("  Methods:")
                    for method in code.methods:
                        print(f"    {method['visibility']} {method['name']}{method['signature']}")
                print()   
        if enums:
            print("\n== ENUMERATIONS ==")
            for code in sorted(enums, key=lambda x: x.name):
                print(f"Enumeration: {code.name} (Confidence: {code.confidence:.2f})")
                if code.definition:
                    print(f"  Definition: {code.definition[:100]}...")
                if code.enum_values:
                    print("  Values:")
                    for value in code.enum_values:
                        print(f"    {value}")
                print()     
        if actors:
            print("\n== ACTORS ==")
            for code in sorted(actors, key=lambda x: x.name):
                print(f"Actor: {code.name} (Confidence: {code.confidence:.2f})")
                if code.definition:
                    print(f"  Definition: {code.definition[:100]}...")
                if code.attributes:
                    print("  Attributes:")
                    for attr in code.attributes:
                        print(f"    {attr['visibility']} {attr['name']}: {attr['type']}")
                if code.methods:
                    print("  Methods:")
                    for method in code.methods:
                        print(f"    {method['visibility']} {method['name']}{method['signature']}")
                print()
    def _ensure_core_entities_present(self):
        for entity_name in self.code_system.core_domain_entities:
            exists=False
            for code in self.code_system.codes.values():
                if code.name.lower()==entity_name.lower():
                    exists=True
                    break
            if exists:continue
            new_id=f"code_{len(self.code_system.codes)+1}"
            new_code=Code(id=new_id,name=entity_name,definition=f"Core entity in {self.domain_name} domain",is_recommendation=True)
            if entity_name in self.code_system.domain_attributes:
                new_code.attributes=self.code_system.domain_attributes[entity_name]
            if entity_name in self.code_system.domain_operations:
                new_code.methods=self.code_system.domain_operations[entity_name]
            if entity_name in self.code_system.domain_actors:
                new_code.stereotypes=["actor"]
            else:
                new_code.stereotypes=["entity"]
            new_code.confidence=0.6
            new_code.notes.append("Added from domain analysis as core entity")
            self.code_system.add_code(new_code)
    def _process_extracted_entities(self,response:str,doc:Document)->bool:
        result=self._extract_json_from_response(response)
        if not result:return False
        entities=result.get("entities",[])
        if not entities:return False
        chunk_id=doc.metadata.get("chunk_id","unknown")
        source=doc.metadata.get("source","unknown")
        added=False
        for entity in entities:
            entity_name=entity.get("name","").strip()
            if not entity_name or self._is_banned_term(entity_name):continue
            name_in_text=self._verify_text_presence(entity_name)
            is_core_entity=entity_name in self.code_system.core_domain_entities
            if not (name_in_text or is_core_entity):continue
            confidence=float(entity.get("confidence",0.5))
            if confidence<0.5:continue
            extracted_text=entity.get("extracted_text","")
            definition=entity.get("definition","")
            entity_type=entity.get("type","Class")
            attributes=[]
            for attr in entity.get("attributes",[]):
                attr_name=attr.get("name","")
                if attr_name:
                    attributes.append({
                        "name":attr_name,
                        "type":attr.get("type","String"),
                        "visibility":attr.get("visibility","+")
                    })
            if not attributes and entity_name in self.code_system.domain_attributes:
                attributes=self.code_system.domain_attributes[entity_name]
            operations=[]
            for op in entity.get("operations",[]):
                op_name=op.get("name","")
                if op_name:
                    operations.append({
                        "name":op_name,
                        "signature":op.get("signature","(): void"),
                        "visibility":op.get("visibility","+")
                    })
            if not operations and entity_name in self.code_system.domain_operations:
                operations=self.code_system.domain_operations[entity_name]
            evidence=entity.get("evidence","")
            enum_values=entity.get("enumValues",[])
            if not enum_values and entity_name in self.code_system.enumerations:
                enum_values=self.code_system.enumerations[entity_name]
            stereotypes=[]
            is_enumeration=False
            is_interface=False
            if entity_type.lower()=="interface":
                stereotypes=["interface"]
                is_interface=True
            elif entity_type.lower()=="enumeration":
                stereotypes=["enumeration"]
                is_enumeration=True
            elif entity_name in self.code_system.domain_actors:
                stereotypes=["actor"]
            else:
                stereotypes=["entity"]
            new_id=f"code_{len(self.code_system.codes)+1}"
            is_recommendation=not name_in_text
            code_obj=Code(id=new_id,name=entity_name,definition=definition,is_recommendation=is_recommendation)
            code_obj.attributes=attributes
            code_obj.methods=operations
            code_obj.stereotypes=stereotypes
            code_obj.evidence_chunks.append(evidence)
            code_obj.evidence_locations.append(f"{source}:{chunk_id}")
            code_obj.confidence=confidence
            code_obj.is_enumeration=is_enumeration
            code_obj.enum_values=enum_values
            code_obj.extracted_text=extracted_text
            code_obj.trace_sources.append(doc.page_content)
            if is_interface:
                code_obj.is_interface=True
            if is_recommendation:
                code_obj.notes.append("Added based on domain analysis")
            self.code_system.add_code(code_obj)
            added=True
        return added
    def _is_banned_term(self,name:str)->bool:
        nl=name.strip().lower()
        if nl in self.banned_words:return True
        if len(nl)<=2:return True
        return False
    def _filter_entities(self):
        to_remove=[]
        for cid,code in self.code_system.codes.items():
            if self._is_banned_term(code.name):
                to_remove.append(cid)
                continue
            if code.confidence<0.4 and not code.is_recommendation:
                to_remove.append(cid)
                continue
        for rid in to_remove:
            if rid in self.code_system.codes:
                del self.code_system.codes[rid]
    def perform_axial_coding(self):
        self._identify_interfaces_and_abstracts()
        self._identify_relationships()
        print(f"Axial Coding complete. Established {len(self.code_system.relationships)} relationships.")
    def _identify_interfaces_and_abstracts(self):
        valid_codes=[]
        for cid,code in self.code_system.codes.items():
            if not self._is_banned_term(code.name):
                valid_codes.append({"id":cid,"name":code.name,"methods":[m["signature"] for m in code.methods],
                                   "attributes":[f"{a['name']}: {a['type']}" for a in code.attributes],
                                   "stereotypes":code.stereotypes,"extracted_text":code.extracted_text})
        if not valid_codes:return
        prompt=f"""
# Interface and Abstract Class Identification
## Current Classes:
{json.dumps(valid_codes[:20],indent=2)}
## Identification Task:
Identify which classes should be interfaces or abstract classes based ONLY on direct evidence in the extracted_text field.
DO NOT infer or guess - only mark as interface/abstract if clearly indicated.
## Output Format:
{{
"interfaces": [
    {{
    "id": "class_id",
    "evidence": "Exact text evidence supporting interface classification"
    }}
],
"abstractClasses": [
    {{
    "id": "class_id",
    "evidence": "Exact text evidence supporting abstract classification"
    }}
]
}}"""
        response=self._call_llm(prompt)
        result=self._extract_json_from_response(response)
        if result:
            for interface in result.get("interfaces",[]):
                class_id=interface.get("id","")
                evidence=interface.get("evidence","")
                if not class_id or not evidence or class_id not in self.code_system.codes:continue
                code=self.code_system.codes[class_id]
                if evidence.lower() in code.extracted_text.lower():
                    code.is_interface=True
                    if "interface" not in code.stereotypes:
                        code.stereotypes.append("interface")
            for abstract in result.get("abstractClasses",[]):
                class_id=abstract.get("id","")
                evidence=abstract.get("evidence","")
                if not class_id or not evidence or class_id not in self.code_system.codes:continue
                code=self.code_system.codes[class_id]
                if evidence.lower() in code.extracted_text.lower():
                    code.is_abstract=True
                    if "abstract" not in code.stereotypes:
                        code.stereotypes.append("abstract")
    def _identify_relationships(self):
        valid_classes=[c for cid,c in self.code_system.codes.items() if not self._is_banned_term(c.name)]
        pairs_to_analyze=[]
        for i,c1 in enumerate(valid_classes):
            for c2 in valid_classes[i+1:]:
                pairs_to_analyze.append((c1.id,c2.id))
        batch_size=5
        for i in range(0,len(pairs_to_analyze),batch_size):
            batch=pairs_to_analyze[i:i+batch_size]
            self._identify_relationship_batch(batch)
    def _identify_relationship_batch(self,id_pairs):
        pair_info=[]
        for source_id,target_id in id_pairs:
            source=self.code_system.codes[source_id]
            target=self.code_system.codes[target_id]
            pair_info.append({
                "source":{"id":source_id,"name":source.name,"extracted_text":source.extracted_text},
                "target":{"id":target_id,"name":target.name,"extracted_text":target.extracted_text}
            })
        domain_rel_hints=""
        for pair in pair_info:
            src_name=self.code_system.codes[pair["source"]["id"]].name
            if src_name in self.code_system.domain_relationships:
                domain_rel_hints+=f"Domain hint: {src_name} has these relationships: {', '.join(self.code_system.domain_relationships[src_name])}\n"
        prompt=f"""
# UML Relationship Analysis
## Class Pairs:
{json.dumps(pair_info,indent=2)}
{domain_rel_hints}
## Analysis Task:
Determine UML relationships between each pair based on textual evidence or domain knowledge.
Look for explicit or implicit relationships in the context.
Consider domain-specific patterns and typical relationships.
Valid relationship types:
- IS_A: Child inherits from parent
- IS_PART_OF: Composition relationship
- IMPLEMENTS: Class implements interface
- DEPENDS_ON: One class requires another
- USES: One class uses another
- ASSOCIATION: General relationship
- MANAGES: One class manages another
- CREATES: One class creates another
- NONE: No relationship found
Guidelines:
- Find relationships based on text evidence when available
- Where text is unclear, apply domain knowledge from the context
- Confidence should reflect evidence strength (0.5-1.0)
- For domain-driven relationships without direct evidence, use confidence 0.6
- If no relationship exists, use NONE
Output JSON:
{{
"relationships": [
    {{
    "sourceId": "source_id",
    "targetId": "target_id",
    "type": "IS_A|IS_PART_OF|IMPLEMENTS|DEPENDS_ON|USES|ASSOCIATION|MANAGES|CREATES|NONE",
    "association_name": "verb phrase (e.g., 'creates', 'manages')",
    "evidence": "Text supporting this relationship",
    "confidence": 0.8,
    "multiplicity": {{ "source": "1|0..1|*", "target": "1|0..1|*" }}
    }}
]
}}"""
        context_chunks=[]
        for source_id,target_id in id_pairs:
            source_name=self.code_system.codes[source_id].name
            target_name=self.code_system.codes[target_id].name
            docs=self._get_evidence_from_query(f"{self.domain_name} {source_name} relationship {target_name}")
            if docs:
                context_chunks.extend(docs[:2])
        if context_chunks:
            context="\n\n".join([doc.page_content for doc in context_chunks[:3]])
            prompt+=f"\n\nRelevant domain context:\n{context}"
        response=self._call_llm(prompt)
        result=self._extract_json_from_response(response)
        if result:
            for rel in result.get("relationships",[]):
                rel_type=rel.get("type","NONE")
                if rel_type=="NONE":continue
                source_id=rel.get("sourceId")
                target_id=rel.get("targetId")
                if not source_id in self.code_system.codes or not target_id in self.code_system.codes:continue
                evidence=rel.get("evidence","")
                confidence=float(rel.get("confidence",0.6))
                if confidence<0.7:continue
                association_name=rel.get("association_name","")
                multiplicity=rel.get("multiplicity",{"source":"1","target":"*"})
                relationship_type=CSLRelationshipType.ASSOCIATION
                if rel_type=="IS_A":relationship_type=CSLRelationshipType.IS_A
                elif rel_type=="IS_PART_OF":relationship_type=CSLRelationshipType.IS_PART_OF
                elif rel_type=="IMPLEMENTS":relationship_type=CSLRelationshipType.IMPLEMENTATION
                elif rel_type=="DEPENDS_ON":relationship_type=CSLRelationshipType.DEPENDS_ON
                elif rel_type=="USES":relationship_type=CSLRelationshipType.USES
                elif rel_type=="MANAGES":relationship_type=CSLRelationshipType.MANAGES
                elif rel_type=="CREATES":relationship_type=CSLRelationshipType.CREATES
                relationship_id=f"rel_{len(self.code_system.relationships)+1}"
                new_rel=CodeRelationship(
                    id=relationship_id,
                    source_code_id=source_id,
                    target_code_id=target_id,
                    relationship_type=relationship_type,
                    association_name=association_name,
                    confidence=confidence,
                    multiplicity=multiplicity
                )
                if evidence:
                    new_rel.evidence_chunks.append(evidence)
                source=self.code_system.codes[source_id]
                target=self.code_system.codes[target_id]
                if relationship_type==CSLRelationshipType.IMPLEMENTATION and not target.is_interface:
                    new_rel.relationship_type=CSLRelationshipType.ASSOCIATION
                    new_rel.association_name="uses"
                self.code_system.add_relationship(new_rel)
    def perform_selective_coding(self):
        self.code_system.calculate_relevance_scores()
        self._fix_relationship_multiplicities()
        self._validate_model_elements()
        self._enrich_class_features()
        self._prune_excessive_relationships()
        self._limit_relationships_per_class(max_relationships=3)
        self._ensure_interface_implementations()
        self._ensure_inheritance_hierarchies()
        print(f"Selective Coding complete. Model contains {len(self.code_system.codes)} classes and {len(self.code_system.relationships)} relationships.")
    def _prune_excessive_relationships(self):
        seen_pairs=set()
        to_remove=[]
        for rel in sorted(self.code_system.relationships,key=lambda r:(self._relationship_type_priority(r.relationship_type),r.confidence),reverse=True):
            pair=tuple(sorted([rel.source_code_id,rel.target_code_id]))
            if pair in seen_pairs:
                to_remove.append(rel)
            else:
                seen_pairs.add(pair)
        for rel in to_remove:
            if rel in self.code_system.relationships:
                self.code_system.relationships.remove(rel)
    def _limit_relationships_per_class(self,max_relationships=3):
        class_rel_counts={}
        for rel in sorted(self.code_system.relationships,key=lambda r:(self._relationship_type_priority(r.relationship_type),r.confidence),reverse=True):
            source_id=rel.source_code_id
            class_rel_counts[source_id]=class_rel_counts.get(source_id,0)+1
        to_remove=[]
        for rel in self.code_system.relationships:
            if class_rel_counts[rel.source_code_id]>max_relationships:
                to_remove.append(rel)
                class_rel_counts[rel.source_code_id]-=1
        for rel in to_remove:
            if rel in self.code_system.relationships:
                self.code_system.relationships.remove(rel)
    def _fix_relationship_multiplicities(self):
        for rel in self.code_system.relationships:
            if rel.relationship_type in [CSLRelationshipType.IS_A,CSLRelationshipType.IMPLEMENTATION]:
                rel.multiplicity={"source":"1","target":"1"}
            elif rel.relationship_type==CSLRelationshipType.IS_PART_OF:
                rel.multiplicity={"source":"1..*","target":"1"}
            else:
                rel.multiplicity={"source":"1","target":"*"}
    def _validate_model_elements(self):
        self._remove_duplicate_relationships()
        self._validate_interfaces()
        self._validate_enumerations()
        self._validate_inheritance()
        self._fix_bidirectional_relationships()
    def _remove_duplicate_relationships(self):
        relationship_map={}
        for rel in self.code_system.relationships:
            key=(rel.source_code_id,rel.target_code_id)
            if key not in relationship_map:
                relationship_map[key]=[]
            relationship_map[key].append(rel)
        to_keep=[]
        for key,relations in relationship_map.items():
            if len(relations)<=1:
                to_keep.extend(relations)
                continue
            relations.sort(key=lambda r:(self._relationship_type_priority(r.relationship_type),r.confidence),reverse=True)
            to_keep.append(relations[0])
        self.code_system.relationships=to_keep
        for code in self.code_system.codes.values():
            code.outgoing_relationships=[r for r in code.outgoing_relationships if r in to_keep]
            code.incoming_relationships=[r for r in code.incoming_relationships if r in to_keep]
    def _relationship_type_priority(self,rel_type:CSLRelationshipType)->int:
        priorities={
            CSLRelationshipType.IS_A:10,
            CSLRelationshipType.IMPLEMENTATION:9,
            CSLRelationshipType.IS_PART_OF:8,
            CSLRelationshipType.CREATES:5,
            CSLRelationshipType.MANAGES:5,
            CSLRelationshipType.USES:3,
            CSLRelationshipType.DEPENDS_ON:3,
            CSLRelationshipType.PERFORMS:2,
            CSLRelationshipType.ASSOCIATION:1,
        }
        return priorities.get(rel_type,0)
    def _validate_interfaces(self):
        for code in self.code_system.codes.values():
            if code.is_interface:
                code.attributes=[]
                if "interface" not in code.stereotypes:
                    code.stereotypes.append("interface")
                if not code.methods:
                    code.methods=self._generate_basic_methods(code.name,is_interface=True)
        to_fix=[]
        for rel in self.code_system.relationships:
            if rel.relationship_type!=CSLRelationshipType.IMPLEMENTATION:continue
            source=self.code_system.codes.get(rel.source_code_id)
            target=self.code_system.codes.get(rel.target_code_id)
            if not source or not target:
                to_fix.append((rel,None))
                continue
            if not target.is_interface:
                to_fix.append((rel,CSLRelationshipType.ASSOCIATION))
                continue
            if "actor" in source.stereotypes:
                to_fix.append((rel,CSLRelationshipType.ASSOCIATION))
                continue
        for rel,new_type in to_fix:
            if new_type is None:
                if rel in self.code_system.relationships:
                    self.code_system.relationships.remove(rel)
            else:
                rel.relationship_type=new_type
                rel.association_name="uses"
                rel.multiplicity={"source":"1","target":"1"}
    def _validate_enumerations(self):
        for code in self.code_system.codes.values():
            if code.is_enumeration:
                code.attributes=[]
                code.methods=[]
                if "enumeration" not in code.stereotypes:
                    code.stereotypes.append("enumeration")
                if not code.enum_values:
                    if code.name in self.code_system.enumerations:
                        code.enum_values=self.code_system.enumerations[code.name]
                    else:
                        code.enum_values=self._generate_enum_values(code.name)
                        code.notes.append("Enum values inferred from name")
    def _generate_enum_values(self,enum_name:str)->List[str]:
        common_enums={
            "status":["ACTIVE","INACTIVE","PENDING","COMPLETED","CANCELED"],
            "type":["STANDARD","PREMIUM","BASIC","CUSTOM"],
            "role":["ADMIN","USER","GUEST","MODERATOR"],
            "priority":["HIGH","MEDIUM","LOW"],
            "state":["NEW","IN_PROGRESS","DONE","CANCELLED"],
        }
        name_lower=enum_name.lower()
        for pattern,values in common_enums.items():
            if pattern in name_lower:
                return values
        prompt=f"""Generate 3-6 enum values for '{enum_name}' in {self.domain_name} domain.
        Return JSON array: ["VALUE1", "VALUE2", "VALUE3"]
        Use UPPER_CASE convention."""
        response=self._call_llm(prompt)
        result=self._extract_json_from_response(response)
        return result if result else ["VALUE1","VALUE2","VALUE3"]
    def _validate_inheritance(self):
        to_fix=[]
        for rel in self.code_system.relationships:
            if rel.relationship_type!=CSLRelationshipType.IS_A:continue
            source=self.code_system.codes.get(rel.source_code_id)
            target=self.code_system.codes.get(rel.target_code_id)
            if not source or not target:
                to_fix.append((rel,None))
                continue
            if "actor" in source.stereotypes or "actor" in target.stereotypes:
                to_fix.append((rel,CSLRelationshipType.ASSOCIATION))
                continue
            if source.is_interface or target.is_interface:
                to_fix.append((rel,CSLRelationshipType.IMPLEMENTATION))
            elif source.is_enumeration or target.is_enumeration:
                to_fix.append((rel,CSLRelationshipType.ASSOCIATION))
            elif self._check_inheritance_cycle(source.id,target.id):
                to_fix.append((rel,CSLRelationshipType.ASSOCIATION))
        for rel,new_type in to_fix:
            if new_type is None:
                if rel in self.code_system.relationships:
                    self.code_system.relationships.remove(rel)
            else:
                rel.relationship_type=new_type
                rel.association_name="relates to" if new_type==CSLRelationshipType.ASSOCIATION else rel.association_name
                rel.multiplicity={"source":"1","target":"1"}
    def _check_inheritance_cycle(self,source_id:str,target_id:str,visited=None)->bool:
        if visited is None:
            visited=set()
        if source_id in visited:return True
        visited.add(source_id)
        for rel in self.code_system.relationships:
            if rel.relationship_type==CSLRelationshipType.IS_A and rel.source_code_id==target_id:
                if rel.target_code_id==source_id:return True
                if self._check_inheritance_cycle(source_id,rel.target_code_id,visited):return True
        return False
    def _fix_bidirectional_relationships(self):
        relationship_pairs={}
        for rel in self.code_system.relationships:
            key=tuple(sorted([rel.source_code_id,rel.target_code_id]))
            if key not in relationship_pairs:
                relationship_pairs[key]=[]
            relationship_pairs[key].append(rel)
        for pair,relations in relationship_pairs.items():
            if len(relations)<=1:continue
            relations.sort(key=lambda r:(self._relationship_type_priority(r.relationship_type),r.confidence),reverse=True)
            to_keep=relations[0]
            to_remove=relations[1:]
            if to_keep.relationship_type in [CSLRelationshipType.ASSOCIATION,CSLRelationshipType.USES]:
                if not to_keep.association_name:
                    for r in to_remove:
                        if r.association_name:
                            to_keep.association_name=r.association_name
                            break
            for rel in to_remove:
                if rel in self.code_system.relationships:
                    self.code_system.relationships.remove(rel)
    def _enrich_class_features(self):
        classes_to_enrich=[]
        for cid,code in self.code_system.codes.items():
            if code.is_enumeration:continue
            if code.is_interface:
                if len(code.methods)<2:
                    classes_to_enrich.append((cid,code))
            else:
                if len(code.attributes)<1 or len(code.methods)<1:
                    classes_to_enrich.append((cid,code))
        batch_size=5
        for i in range(0,len(classes_to_enrich),batch_size):
            batch=classes_to_enrich[i:i+batch_size]
            self._enrich_class_batch(batch)
    def _enrich_class_batch(self,class_batch):
        class_info=[]
        for cid,code in class_batch:
            info={
                "id":cid,
                "name":code.name,
                "definition":code.definition,
                "type":"Interface" if code.is_interface else "AbstractClass" if code.is_abstract else "Class",
                "currentAttributes":[f"{a['visibility']}{a['name']}: {a['type']}" for a in code.attributes],
                "currentMethods":[f"{m['visibility']}{m['name']}{m['signature']}" for m in code.methods],
            }
            class_info.append(info)
        prompt=f"""
# Enrich UML Class Features for {self.domain_name}
## Classes to Enrich:
{json.dumps(class_info,indent=2)}
## Enrichment Task:
Provide domain-specific attributes and operations for each class.
Guidelines:
- Attributes: Include name, type (String, Integer, etc.), visibility (+, -, #)
- Operations: Include name, signature with parameters/return type, visibility
- Interfaces: Only methods, no attributes
- Ensure domain relevance
- Use camelCase for methods, lowerCamelCase for attributes
Output JSON:
{{
"enrichedClasses": [
    {{
    "id": "class_id",
    "attributes": [{{"name": "attrName", "type": "Type", "visibility": "+|-|#"}}],
    "operations": [{{"name": "opName", "signature": "(param: Type): ReturnType", "visibility": "+|-|#"}}]
    }}
]
}}"""
        context_docs=[]
        for cid,code in class_batch:
            docs=self._get_evidence_from_query(f"{self.domain_name} {code.name} attributes methods")
            if docs:
                context_docs.extend(docs)
        if context_docs:
            context="\n\n".join([doc.page_content for doc in context_docs[:3]])
            prompt+=f"\n\nAdditional domain context:\n{context}"
        response=self._call_llm(prompt)
        result=self._extract_json_from_response(response)
        if result:
            for enriched in result.get("enrichedClasses",[]):
                class_id=enriched.get("id")
                if not class_id or class_id not in self.code_system.codes:continue
                code_obj=self.code_system.codes[class_id]
                if not code_obj.is_interface:
                    attributes=[{
                        "name":a.get("name",""),
                        "type":a.get("type","String"),
                        "visibility":a.get("visibility","+"),
                    } for a in enriched.get("attributes",[])]
                    if attributes:
                        code_obj.attributes=attributes
                        code_obj.notes.append("Attributes enriched based on domain analysis")
                operations=[{
                    "name":o.get("name",""),
                    "signature":o.get("signature","(): void"),
                    "visibility":o.get("visibility","+"),
                } for o in enriched.get("operations",[])]
                if operations:
                    code_obj.methods=operations
                    code_obj.notes.append("Methods enriched based on domain analysis")
        else:
            for cid,code in class_batch:
                if not code.is_interface and not code.attributes:
                    code.attributes=self._generate_basic_attributes(code.name)
                    code.notes.append("Basic attributes generated")
                if not code.methods:
                    code.methods=self._generate_basic_methods(code.name,code.is_interface)
                    code.notes.append("Basic methods generated")
    def _generate_basic_attributes(self,class_name:str)->List[Dict[str,str]]:
        attributes=[
            {"name":"id","type":"String","visibility":"+"},
            {"name":"name","type":"String","visibility":"+"},
        ]
        name_lower=class_name.lower()
        if "date" in name_lower:
            attributes.append({"name":"date","type":"Date","visibility":"+"})
        if "user" in name_lower or "account" in name_lower:
            attributes.append({"name":"email","type":"String","visibility":"-"})
        return attributes
    def _generate_basic_methods(self,class_name:str,is_interface:bool=False)->List[Dict[str,str]]:
        methods=[
            {"name":"getId","signature":"(): String","visibility":"+"},
            {"name":"getName","signature":"(): String","visibility":"+"},
        ]
        name_lower=class_name.lower()
        if name_lower.endswith("repository"):
            entity=class_name[:-10]
            methods.extend([
                {"name":f"find{entity}ById","signature":"(id: String): "+entity,"visibility":"+"},
                {"name":"findAll","signature":f"(): List<{entity}>","visibility":"+"},
            ])
        elif name_lower.endswith("service"):
            entity=class_name[:-7]
            methods.extend([
                {"name":f"create{entity}","signature":f"({entity.lower()}: {entity}): {entity}","visibility":"+"},
                {"name":f"update{entity}","signature":f"({entity.lower()}: {entity}): Boolean","visibility":"+"},
            ])
        return methods
    def _ensure_interface_implementations(self):
        interfaces=[code for _,code in self.code_system.codes.items() if code.is_interface]
        for interface in interfaces:
            has_implementation=False
            for rel in self.code_system.relationships:
                if rel.target_code_id==interface.id and rel.relationship_type==CSLRelationshipType.IMPLEMENTATION:
                    has_implementation=True
                    break
            if has_implementation:continue
            implementer_candidates=[]
            for cid,code in self.code_system.codes.items():
                if code.is_interface or code.is_enumeration or "actor" in code.stereotypes:continue
                similarity=SequenceMatcher(None,interface.name.lower(),code.name.lower()).ratio()
                if similarity>0.8:
                    implementer_candidates.append((cid,3))
                elif similarity>0.5:
                    implementer_candidates.append((cid,2))
                else:
                    implementer_candidates.append((cid,1))
            if implementer_candidates:
                implementer_candidates.sort(key=lambda x:x[1],reverse=True)
                implementer_id=implementer_candidates[0][0]
                rel_id=f"rel_{len(self.code_system.relationships)+1}"
                new_rel=CodeRelationship(
                    id=rel_id,
                    source_code_id=implementer_id,
                    target_code_id=interface.id,
                    relationship_type=CSLRelationshipType.IMPLEMENTATION,
                    association_name="implements",
                    confidence=0.7,
                    multiplicity={"source":"1","target":"1"},
                )
                self.code_system.add_relationship(new_rel)
                self.code_system.codes[implementer_id].notes.append(f"Added implementation of {interface.name}")
    def _ensure_inheritance_hierarchies(self):
        common_base_classes=["Entity","Resource","Record","Data"]
        if hasattr(self,"common_base_classes"):
            common_base_classes=self.common_base_classes
        for actor in self.code_system.domain_actors:
            if actor not in common_base_classes:
                common_base_classes.append(actor)
        has_parent=set()
        for rel in self.code_system.relationships:
            if rel.relationship_type==CSLRelationshipType.IS_A:
                has_parent.add(rel.source_code_id)
        for cid,code in self.code_system.codes.items():
            if (code.is_enumeration or code.is_interface or "actor" in code.stereotypes or
                cid in has_parent):continue
            candidates=[]
            for parent_id,parent in self.code_system.codes.items():
                if (parent_id==cid or parent.is_interface or parent.is_enumeration or
                    "actor" in parent.stereotypes):continue
                score=self._inheritance_compatibility_score(code,parent)
                if score>=2:
                    candidates.append((parent_id,score))
            if candidates:
                candidates.sort(key=lambda x:x[1],reverse=True)
                best_parent_id=candidates[0][0]
                rel_id=f"rel_{len(self.code_system.relationships)+1}"
                new_rel=CodeRelationship(
                    id=rel_id,
                    source_code_id=cid,
                    target_code_id=best_parent_id,
                    relationship_type=CSLRelationshipType.IS_A,
                    association_name="extends",
                    confidence=0.7,
                    multiplicity={"source":"1","target":"1"},
                )
                if self.code_system.add_relationship(new_rel):
                    code.notes.append(f"Added inheritance from {self.code_system.codes[best_parent_id].name}")
    def _inheritance_compatibility_score(self,child:Code,parent:Code)->float:
        score=0.0
        child_name=child.name.lower()
        parent_name=parent.name.lower()
        similarity=SequenceMatcher(None,child_name,parent_name).ratio()
        score+=similarity*3
        if parent_name in child_name and parent_name!=child_name:
            score+=2
        child_attrs=set(a["name"] for a in child.attributes)
        parent_attrs=set(a["name"] for a in parent.attributes)
        common_attrs=len(child_attrs.intersection(parent_attrs))
        score+=common_attrs*0.5
        if parent.is_abstract:
            score+=1
        common_base_classes=["Entity","Resource","Record","Data"]
        if hasattr(self,"common_base_classes"):
            common_base_classes=self.common_base_classes
        if parent_name in [n.lower() for n in common_base_classes]:
            score+=1
        return score
    def generate_plantuml(self)->str:
        def sanitize(name):
            result=re.sub(r"\s+","_",name)
            result=re.sub(r"[^a-zA-Z0-9_]","",result)
            if result and result[0].isdigit():
                result="_"+result
            return result if result else "Item"
        lines=["@startuml",f"' {self.domain_name} Domain Model","skinparam monochrome true",
              "skinparam classAttributeIconSize 0","skinparam packageStyle rectangle",
              "skinparam shadowing false","skinparam class {","  BackgroundColor white",
              "  ArrowColor black","  BorderColor black","  FontSize 12","}",""]
        stereotype_map={"interface":"<<interface>>","abstract":"<<abstract>>",
                       "class":"<<class>>","enumeration":"<<enumeration>>","actor":"<<actor>>"}
        actors=[]
        interfaces=[]
        abstract_classes=[]
        enumerations=[]
        concrete_classes=[]
        for code in self.code_system.codes.values():
            if self._is_banned_term(code.name):continue
            if "actor" in code.stereotypes:
                actors.append(code)
            elif code.is_enumeration:
                enumerations.append(code)
            elif code.is_interface:
                interfaces.append(code)
            elif code.is_abstract:
                abstract_classes.append(code)
            else:
                concrete_classes.append(code)
        if actors:
            lines.append("package Actors {")
            for actor in actors:
                stereotypes=[stereotype_map[s] for s in actor.stereotypes if s in stereotype_map]
                stereotype_str=" ".join(stereotypes) if stereotypes else ""
                lines.append(f"  class {sanitize(actor.name)} {stereotype_str} {{")
                for attr in actor.attributes:
                    lines.append(f"    {attr['visibility']} {attr['name']}: {attr['type']}")
                for method in actor.methods:
                    lines.append(f"    {method['visibility']} {method['name']}{method['signature']}")
                lines.append("  }")
            lines.append("}")
        if interfaces:
            lines.append("package Interfaces {")
            for interface in interfaces:
                stereotypes=[stereotype_map[s] for s in interface.stereotypes if s in stereotype_map]
                stereotype_str=" ".join(stereotypes) if stereotypes else ""
                lines.append(f"  interface {sanitize(interface.name)} {stereotype_str} {{")
                for method in interface.methods:
                    lines.append(f"    {method['visibility']} {method['name']}{method['signature']}")
                lines.append("  }")
            lines.append("}")
        if abstract_classes:
            lines.append("package AbstractClasses {")
            for abstract in abstract_classes:
                stereotypes=[stereotype_map[s] for s in abstract.stereotypes if s in stereotype_map]
                stereotype_str=" ".join(stereotypes) if stereotypes else ""
                lines.append(f"  abstract class {sanitize(abstract.name)} {stereotype_str} {{")
                for attr in abstract.attributes:
                    lines.append(f"    {attr['visibility']} {attr['name']}: {attr['type']}")
                for method in abstract.methods:
                    lines.append(f"    {method['visibility']} {method['name']}{method['signature']}")
                lines.append("  }")
            lines.append("}")
        if enumerations:
            lines.append("package Enumerations {")
            for enum in enumerations:
                stereotypes=[stereotype_map[s] for s in enum.stereotypes if s in stereotype_map]
                stereotype_str=" ".join(stereotypes) if stereotypes else ""
                lines.append(f"  enum {sanitize(enum.name)} {stereotype_str} {{")
                for value in enum.enum_values:
                    lines.append(f"    {value}")
                lines.append("  }")
            lines.append("}")
        if concrete_classes:
            lines.append("package Entities {")
            for concrete in concrete_classes:
                stereotypes=[stereotype_map[s] for s in concrete.stereotypes if s in stereotype_map]
                stereotype_str=" ".join(stereotypes) if stereotypes else ""
                lines.append(f"  class {sanitize(concrete.name)} {stereotype_str} {{")
                for attr in concrete.attributes:
                    lines.append(f"    {attr['visibility']} {attr['name']}: {attr['type']}")
                for method in concrete.methods:
                    lines.append(f"    {method['visibility']} {method['name']}{method['signature']}")
                lines.append("  }")
            lines.append("}")
        lines.append("")
        for rel in self.code_system.relationships:
            source=self.code_system.codes.get(rel.source_code_id)
            target=self.code_system.codes.get(rel.target_code_id)
            if not source or not target:continue
            source_name=sanitize(source.name)
            target_name=sanitize(target.name)
            if rel.relationship_type==CSLRelationshipType.IS_A:
                lines.append(f"{source_name} --|> {target_name}")
            elif rel.relationship_type==CSLRelationshipType.IMPLEMENTATION:
                lines.append(f"{source_name} ..|> {target_name}")
            elif rel.relationship_type==CSLRelationshipType.IS_PART_OF:
                lines.append(f"{source_name} *-- {target_name} : {rel.association_name or 'contains'}")
            elif rel.relationship_type in [CSLRelationshipType.DEPENDS_ON,CSLRelationshipType.USES]:
                lines.append(f"{source_name} ..> {target_name} : {rel.association_name or 'uses'}")
            elif rel.relationship_type==CSLRelationshipType.MANAGES:
                source_mult=rel.multiplicity.get("source","1")
                target_mult=rel.multiplicity.get("target","*")
                lines.append(f'{source_name} "{source_mult}" --> "{target_mult}" {target_name} : manages')
            elif rel.relationship_type==CSLRelationshipType.CREATES:
                source_mult=rel.multiplicity.get("source","1")
                target_mult=rel.multiplicity.get("target","*")
                lines.append(f'{source_name} "{source_mult}" --> "{target_mult}" {target_name} : creates')
            else:
                source_mult=rel.multiplicity.get("source","1")
                target_mult=rel.multiplicity.get("target","*")
                lines.append(f'{source_name} "{source_mult}" -- "{target_mult}" {target_name} : {rel.association_name or ""}')
        lines.append("")
        lines.append("note as ModelSource")
        lines.append(f"  Domain model for {self.domain_name}")
        lines.append("  Generated from requirements using AI-driven analysis")
        lines.append(f"  Created on {time.strftime('%Y-%m-%d')}")
        lines.append("end note")
        lines.append("@enduml")
        return "\n".join(lines)
    def generate_traceability_report(self)->str:
        report=[f"# Traceability Report for {self.domain_name} Domain Model\n"]
        report.append(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("## Domain Description")
        report.append(self.code_system.domain_description)
        report.append("\n")
        report.append("## Core Entities")
        report.append(", ".join(self.code_system.core_domain_entities))
        report.append("\n")
        report.append("## Classes and Elements")
        for code in sorted(self.code_system.codes.values(),key=lambda x:x.name):
            if self._is_banned_term(code.name):continue
            report.append(f"### {code.name}")
            report.append(f"- **Type:** {', '.join(code.stereotypes)}")
            report.append(f"- **Definition:** {code.definition}")
            if code.attributes:
                report.append("- **Attributes:**")
                for attr in code.attributes:
                    report.append(f"  - {attr['visibility']} {attr['name']}: {attr['type']}")
            if code.methods:
                report.append("- **Methods:**")
                for method in code.methods:
                    report.append(f"  - {method['visibility']} {method['name']}{method['signature']}")
            if code.enum_values:
                report.append("- **Enum Values:**")
                for value in code.enum_values:
                    report.append(f"  - {value}")
            report.append("- **Evidence Locations:**")
            for loc in code.evidence_locations:
                report.append(f"  - {loc}")
            report.append("- **Confidence:** {:.2f}".format(code.confidence))
            if code.is_recommendation:
                report.append("- **Note:** Added based on domain analysis")
            if code.notes:
                report.append("- **Notes:**")
                for note in code.notes:
                    report.append(f"  - {note}")
            report.append("\n")
        report.append("## Relationships")
        for rel in sorted(self.code_system.relationships,key=lambda r:r.id):
            source=self.code_system.codes.get(rel.source_code_id)
            target=self.code_system.codes.get(rel.target_code_id)
            if not source or not target:continue
            report.append(f"### {source.name} -> {target.name}")
            report.append(f"- **Type:** {rel.relationship_type.value}")
            report.append(f"- **Name:** {rel.association_name}")
            report.append(f"- **Multiplicity:** {rel.multiplicity['source']} : {rel.multiplicity['target']}")
            report.append("- **Evidence:**")
            for chunk in rel.evidence_chunks:
                report.append(f"  - {chunk}")
            report.append("- **Confidence:** {:.2f}".format(rel.confidence))
            report.append("\n")
        return "\n".join(report)
    def save_output(self,plantuml_code:str,report:str,output_dir:str="output"):
        os.makedirs(output_dir,exist_ok=True)
        timestamp=time.strftime("%Y%m%d_%H%M%S")
        plantuml_file=os.path.join(output_dir,f"{self.domain_name.lower()}_model_{timestamp}.puml")
        with open(plantuml_file,"w",encoding="utf-8") as f:
            f.write(plantuml_code)
        print(f"PlantUML diagram saved to {plantuml_file}")
        report_file=os.path.join(output_dir,f"{self.domain_name.lower()}_report_{timestamp}.md")
        with open(report_file,"w",encoding="utf-8") as f:
            f.write(report)
        print(f"Traceability report saved to {report_file}")
    def execute(self,file_path:str,domain_description:str,output_dir:str="output"):
        self.setup(file_path,domain_description)
        plantuml_code,report=self.run_pipeline()
        self.save_output(plantuml_code,report,output_dir)
        return plantuml_code,report
def main():
    parser=argparse.ArgumentParser(description="Domain Modeling Pipeline")
    parser.add_argument("--file",type=str,required=True,help="Path to the requirements file")
    parser.add_argument("--description",type=str,required=True,help="Domain description")
    parser.add_argument("--api-key",type=str,required=True,help="OpenAI API key")
    parser.add_argument("--output-dir",type=str,default="output",help="Output directory")
    parser.add_argument("--model",type=str,default="gpt-4.1-mini",help="LLM model name")
    args=parser.parse_args()
    pipeline=DomainModelingPipeline(api_key=args.api_key,model_name=args.model)
    pipeline.execute(args.file,args.description,args.output_dir)
if __name__=="__main__":
    main()
