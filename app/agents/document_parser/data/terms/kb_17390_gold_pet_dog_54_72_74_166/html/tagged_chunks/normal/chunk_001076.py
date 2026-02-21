from langchain_core.documents import Document

chunk = Document(
    page_content=('무지개다리위로금보장개시일이라 합니다) 이후에 사망한<br>경우 이 특별약관의 보험가입금액을 무지개다리위로금(강아지, 사망)으로 '
 '보험수<br>익자에게 지급합니다.<br>\uf000 제1항의 사망은 동물병원에서 적법하게 시행된 안락사를 포함합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001076',
              'chunk_char_len': 130,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
