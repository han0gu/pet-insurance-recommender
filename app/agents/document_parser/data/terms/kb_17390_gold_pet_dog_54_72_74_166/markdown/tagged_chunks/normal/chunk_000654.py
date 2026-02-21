from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 보험의 목적이 다수인 경우 제1항 내지 제2항은 보험의 목적별로 각각 적용합 해\n'
 '니다. 및# 제7조(특별약관의 자동갱신)\uf000 이 특별약관의 【갱신계약】은 "제도성 특별약관 - 보장특약 자동갱신(추가납입\n'
 '형) 특별약관"에 의해 계약자의 선택에 따라 자동갱신으로 운영합니다.\n'
 '\uf000 제1항에 의해 자동갱신을 적용할 경우 보험증권에 그 내용을 기재하여 드립니다. 반\n'
 '려동-'),
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
 'indexing': {'chunk_id': 'chunk_000654',
              'chunk_char_len': 204,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
