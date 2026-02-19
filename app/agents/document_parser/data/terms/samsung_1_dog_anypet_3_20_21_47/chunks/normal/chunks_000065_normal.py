from langchain_core.documents import Document

chunk = Document(
    page_content=('【타인을 위한 계약】 계약자가 다른 사람의 이익을 위하여 자기의 이름으로 체결하는 보험계약을 말합니 다.\n'
 '제5관 보험료의 납입\n'
 '제21 조(제1회 보험료 등 및 회사의 보장개시)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 13},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000065',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
