from langchain_core.documents import Document

chunk = Document(
    page_content=('하며 이후 한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관| 에서 | 보장하는 환경성질환 해당 여부를 판단합니다. '
 '|  |\n'
 '| --- | --- | --- |\n'
 '| 구분 | 대상이 되는 질병 | 분류번호 |\n'
 '| 아토피 알레르기성 | 아토피성 피부염 | L20 |\n'
 '| 천 | 혈관운동성 및 알레르기성 비염 비염 | J30 |\n'
 '| 천 | 천식 식 | J45 |\n'
 '|  | 천식지속 상태 | J46 |\n'
 '| 급성 급성 기관지염 | J20 |  |\n'
 '| 폐 | 기관지염 급성 세기관지염 | J21 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_001017',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
