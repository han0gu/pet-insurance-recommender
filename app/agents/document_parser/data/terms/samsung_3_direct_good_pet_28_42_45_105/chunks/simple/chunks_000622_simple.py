from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 계약자는 즉시 갱신계약 보험료를 납입하여야 합니다. 만약, 이 보험료를 납 입하지 않으면 회사는 지급할 보험금에서 이를 '
 '차감할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 98},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000622',
              'chunk_char_len': 85,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
