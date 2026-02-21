from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '|  | 805 피부질환 |  |\n'
 '| 피부진균감염증(말라세치아, 사상균 등) | 805 피부질환 |  |\n'
 '| 내이염 | 805 피부질환 |  |\n'
 '| 외이염, 외이도염 | 805 피부질환 |  |\n'
 '| 중이염 | 805 피부질환 |  |\n'
 '| 개선충 감염(옴감염) | 805 피부질환 |  |\n'
 '| 벼룩 감염 | 805 피부질환 |  |\n'
 '| 진드기 감염 비만세포종 | 805 피부질환 |  |\n'
 '| 조직구종 | 805 피부질환 |  |\n'
 '| 흑색종 | 805 피부질환 |  |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'skin']},
 'indexing': {'chunk_id': 'chunk_001041',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
