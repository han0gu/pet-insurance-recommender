from langchain_core.documents import Document

chunk = Document(
    page_content=('| 조직구종 | 805 피부질환 |  |\n'
 '| 흑색종 | 805 피부질환 |  |\n'
 '| 각화이상 | 805 피부질환 |  |\n'
 '| 농피증 | 805 피부질환 |  |\n'
 '| 다리 부위 피부염 | 805 피부질환 |  |\n'
 '| 두드러기 | 805 피부질환 |  |\n'
 '| 면역매개성피부질환 | 805 피부질환 |  |\n'
 '| 모낭염 발톱 질환 | 805 피부질환 |  |\n'
 '| 아토피성 피부염 | 805 피부질환 |  |\n'
 '| 알러지성 피부염 | 805 피부질환 |  |\n'
 '| 지간 피부염 | 805 피부질환 |  |\n'
 '| 지루 | 805 피부질환 |  |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_001042',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
