from langchain_core.documents import Document

chunk = Document(
    page_content=('표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관에서 보장하는| 호흡기관련질병 해당 여부를 판단합니다. |  |\n'
 '| --- | --- |\n'
 '| 대상이 되는 항목 | 분류번호 |\n'
 '| 급성상기도감염 | J00~J06 |\n'
 '| 상기도의 상세불명 질환 급성인지 만성인지 명시되지 않은 기관지염 | J39.9 J40 |\n'
 '| 단순성 및 점액화농성 만성기관지염 | J41 |\n'
 '| 상세불명의 만성 기관지염 | J42 |\n'
 '| 천식, 천식지속 상태 | J45, J46 |\n'
 '| 폐렴 | J12~J18 |'),
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
 'indexing': {'chunk_id': 'chunk_001013',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
