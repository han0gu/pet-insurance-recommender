from langchain_core.documents import Document

chunk = Document(
    page_content=("위하여 체결됩니다.</p><br><h1 id='155' style='font-size:14px'>제2조(용어의</h1><br><p "
 "id='156' data-category='paragraph' style='font-size:14px'>정의)</p><br><p "
 "id='157' data-category='paragraph' style='font-size:14px'>이 특별약관에서 사용되는 용어의 "
 "정의는, 이 특별약관의 다른 조항에서 달리 정의<br>되지 않는 한 다음과 같습니다.</p><br><h1 id='158'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001138',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
