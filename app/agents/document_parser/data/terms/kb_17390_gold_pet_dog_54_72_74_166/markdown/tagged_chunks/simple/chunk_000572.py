from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 보험증권에 기재된 반려동물에게 이 특별약관의 보험기간 중 반려동물주요\n'
 '치료에 대한 보장개시일(이하 반려동물주요치료보장개시일이라 합니다) 이후에 "\n'
 '치료구분별 대상원인"(이하 사고라 합니다)이 발생하여 그 치료를 직접적인 목적\n'
 '으로 국내에서 수의사에게 "반려동물주요치료"를 받은 경우에는 치료구분별로 각\n'
 '각의 지급방식에 따라 당일 피보험자가 부담한 반려동물의 치료에 사용된 비용(각\n'
 '종 할인 및 감면, 사후환급금액 등을 제외한 실수납액을 의미합니다. 이하 의료비'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000572',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
