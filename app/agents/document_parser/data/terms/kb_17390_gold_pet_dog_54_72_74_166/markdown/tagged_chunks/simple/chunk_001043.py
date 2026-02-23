from langchain_core.documents import Document

chunk = Document(
    page_content=('| 지간 피부염 | 805 피부질환 |  |\n'
 '| 지루 | 805 피부질환 |  |\n'
 '| 지방조직염 | 805 피부질환 |  |\n'
 '| 피하 농양 / 봉와직염 호산구성 육아종 | 805 피부질환 |  |\n'
 '| 기타 세균성 피부염 | 805 피부질환 |  |\n'
 '| 기타 피부염 | 805 피부질환 |  |\n'
 '| 기타 피부질환 | 805 피부질환 |  |\n'
 '| 기타 선천성 피부질환 | 805 피부질환 |  |\n'
 '| 원인 불명의 귀 소양감 | 805 피부질환 |  |\n'
 '| 탈모 | 805 피부질환 |  |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_001043',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
