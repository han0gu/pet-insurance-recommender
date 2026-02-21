from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 2) | ‘코의 후각기능을 완전히 잃었을 때’라 함은 후각신경의 손상으로 양쪽 코의 후각기능을 완전히 잃은 경우를 말하며, 후각감퇴는 '
 '장해의 대상으 로 하지 않는다. |\n'
 '| 3) | 양쪽 코의 후각기능은 후각인지검사, 후각역치검사 등을 통해 6개월 이 상 고정된 후각의 완전손실이 확인되어야 한다. |\n'
 '| 4) | 코의 추상(추한 모습)장해를 수반한 때에는 기능장해의 지급률과 추상장 해의 지급률을 합산한다. |\n'
 '- \n'
 '- 142 -4. 씹어먹거나# 말하는 장해| 가. 장해의 분류 |  |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000855',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
