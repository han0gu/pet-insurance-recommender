from langchain_core.documents import Document

chunk = Document(
    page_content=('몸통)를 유합(아물어 붙음) 또는 고정한 상태\n'
 '나) 머리뼈(두개골), 제1경추, 제2경추를 모두 유합 또는 고정한 상태\n'
 '7) 뚜렷한 운동장해란 다음 중 어느 하나에 해당하는 경우를 말한다.\n'
 '가) 척추체(척추뼈 몸통)에 골절 또는 탈구로 3개의 척추체(척추뼈 몸 특별통)를 유합(아물어 붙음) 또는 고정한 상태 약\n'
 '나) 머리뼈(두개골)와 제1경추 또는 제1경추와 제2경추를 유합 또는 고 관\n'
 '정한 상태\n'
 '다) 머리뼈(두개골)와 상위목뼈(상위경추: 제1, 2경추) 사이에 CT 검사'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000879',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
