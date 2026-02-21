from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 피보험자가 손해배상을 함으로써 대위 취득하는 것이 있을 경우에는 그 대위권\n'
 '- \uf000 계약자 또는 피보험자는 제1항에 따라 회사가 취득한 권리를 행사하거나 지키는\n'
 '- 것에 관하여 필요한 조치를 하여야 하며 또한 회사가 요구하는 증거나 서류를 제\n'
 '- 출하여야 합니다.\n'
 '- \uf000 제1항 및 제2항에도 불구하고 타인을 위한 계약의 경우에는 회사는 계약자에 대\n'
 '- 한 대위권을 포기합니다.\n'
 '- \uf000 회사는 제1항에 따른 권리가 계약자 또는 피보험자와 생계를 같이 하는 가족에 대'),
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
 'indexing': {'chunk_id': 'chunk_000694',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
