from langchain_core.documents import Document

chunk = Document(
    page_content=('여 중 요한 사항을 고지하지 아니하거나 부실의 고지를 한 때에는 보험자는 그 사실을 안 날로부터 1월내에, 계약을 체결한 날로부터 '
 '3년내에 한하여 계약을 해지할 수 있다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001208',
              'chunk_char_len': 94,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
