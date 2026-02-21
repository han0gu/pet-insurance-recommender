from langchain_core.documents import Document

chunk = Document(
    page_content=('병을 말하며 이후 한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 사항| 이 보장하는 2대호흡계특정질환 해당 여부를 '
 '판단합니다. | 이 보장하는 2대호흡계특정질환 해당 여부를 판단합니다. | 약관에서 |\n'
 '| --- | --- | --- |\n'
 '| 간질영향 호흡기질환 | 대상이 되는 항목 | 분류번호 |\n'
 '| 간질영향 호흡기질환 | 성인호흡곤란증후군 | J80 보 |\n'
 '| 간질영향 호흡기질환 | 폐부종 | J81 |\n'
 '| 간질영향 호흡기질환 | 달리 분류되지 않은 폐호산구증가 | 통약 J82 |'),
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
 'indexing': {'chunk_id': 'chunk_001002',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
