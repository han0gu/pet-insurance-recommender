from langchain_core.documents import Document

chunk = Document(
    page_content=("id='84' data-category='paragraph' style='font-size:14px'>다수인 경우 제1항 내지 제2항은 "
 "보험의 목적별로 각각 적용합니다.</p><br><h1 id='85' style='font-size:14px'>\uf000 보험의 "
 "목적이</h1><br><p id='86' data-category='list' style='font-size:14px'>제6조(특별약관의 "
 '자동갱신)<br>\uf000 이 특별약관의 【갱신계약】은 "제도성 특별약관 - 보장특약 자동갱신(추가납입<br>형) 특별약관"에 의해 '
 '계약자의 선택에'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001097',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
