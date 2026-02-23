from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 타인을 위한 계약의 경우에는 계약자는 그 타인의 동의 특<br>를 얻거나 보험증권을 소지한 경우에 한하여 특별약관을 해지할 수 '
 '있습니다'),
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
 'indexing': {'chunk_id': 'chunk_000907',
              'chunk_char_len': 81,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
