from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 특별약관에서 보장하지 않는 사유로 사망하였을 경우<br>2. 이 특별약관에서 보장하는 사유로 사망하였으나 제1조(보험금의 '
 "지급사유)에<br>서 정한 반려동물 장례서비스를 이용하지 않은 경우 상</p><br><p id='131' "
 "data-category='paragraph' style='font-size:14px'>\uf000 보험의 목적이 다수인 경우 제1항 "
 '내지 제2항은 보험의 목적별로 각각 적용합 해<br>니다'),
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
 'indexing': {'chunk_id': 'chunk_001127',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
