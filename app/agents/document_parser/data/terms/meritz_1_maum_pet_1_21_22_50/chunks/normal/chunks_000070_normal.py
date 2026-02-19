from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑧ 제28조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))에 따라 이 계약이 부 활이 이루어진 경우에는 부활계약을 제2항의 '
 '최초계약으로 봅니다.(부활(효력회복)이 여러차례 발생된 경우에는 각각의 부활(효력회복)계약을 최초계약으로 봅니다)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 11},
 'term_type': 'basic',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000070',
              'chunk_char_len': 140,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
