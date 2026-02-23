from langchain_core.documents import Document

chunk = Document(
    page_content=('| 탈모 | 805 피부질환 |  |\n'
 '| 원인 불명의 피부 소양감 | 805 피부질환 |  |\n'
 '166 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)- 166 -'),
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
 'indexing': {'chunk_id': 'chunk_001044',
              'chunk_char_len': 92,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
