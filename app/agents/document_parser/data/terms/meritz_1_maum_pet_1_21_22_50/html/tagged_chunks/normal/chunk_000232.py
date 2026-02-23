from langchain_core.documents import Document

chunk = Document(
    page_content=("관계)</h1><br><p id='54' data-category='list' style='font-size:14px'>① 회사는 이 "
 '특별약관에 의하여 보상하여야 하는 금액이 의무보험에서 보상하는 금액을<br>초과할 때에 한하여 그 초과액만을 보상합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000232',
              'chunk_char_len': 141,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
