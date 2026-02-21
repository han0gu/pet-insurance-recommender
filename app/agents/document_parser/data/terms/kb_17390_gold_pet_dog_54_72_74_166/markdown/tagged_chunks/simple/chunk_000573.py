from langchain_core.documents import Document

chunk = Document(
    page_content=('종 할인 및 감면, 사후환급금액 등을 제외한 실수납액을 의미합니다. 이하 의료비\n'
 '라 합니다)에서 반려동물의료비보험금 및 보험증권에 기재된 자기부담금을 제외\n'
 '한 금액을 제3항에 따라 보험수익자에게 주요치료보험금으로 지급합니다.\n'
 '< 보험가입금액 1,000만원 기준 >| 치 료 구 분 | 치 료 구 분 | 지급한도 | 치료구분별 보상한도액 |\n'
 '| --- | --- | --- | --- |\n'
 '| MRI/CT | MRI/CT | 연간1회한 | 100만원 |\n'
 '| 백내장/녹내장수술 | 백내장/녹내장수술 | 연간1회한 | 50만원 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000573',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
