from langchain_core.documents import Document

chunk = Document(
    page_content=('| 고급형Ⅱ | 입원 | 1일당 30만원 한도 1일당 | 300만원 한도 | 2,000만원 려동 |\n'
 '| 고급형Ⅱ | 통원 | 1일당 30만원 한도 1일당 | 300만원 한도 | 2,000만원 물 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000547',
              'chunk_char_len': 110,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
