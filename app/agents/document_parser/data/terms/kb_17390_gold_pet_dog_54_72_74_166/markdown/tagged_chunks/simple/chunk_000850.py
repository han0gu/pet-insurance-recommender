from langchain_core.documents import Document

chunk = Document(
    page_content=('- 손된 경우를 말한다.\n'
 '- 2) 귓바퀴의 연골부가 1/2 미만 결손이고 청력에 이상이 없으면 외모의 추\n'
 '상(추한 모습)장해로만 평가한다.라. 평형기능의 장해\n'
 '1) ‘평형기능에 장해를 남긴 때’라 함은 전정기관 이상으로 보행 등 일상\n'
 '생활이 어려운 상태로 아래의 평형장해 평가항목별 합산점수가 30점 이| 상인 | 경우를 말한다. |  |\n'
 '| --- | --- | --- |\n'
 '|  | 항목 내 용 검사 | 점수 |\n'
 '| 양측 전정기능 소실 | 14 |  |\n'
 '| 양측 전정기능 감소 소견 | 10 |  |'),
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
 'indexing': {'chunk_id': 'chunk_000850',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
