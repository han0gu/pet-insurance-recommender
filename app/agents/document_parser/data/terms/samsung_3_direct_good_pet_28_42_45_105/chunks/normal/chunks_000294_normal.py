from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 회사는 경과기간별 해약환급금에 관한 표를 계약자에게 제공하여 드립니다. ⑤ 제32조의2(위법계약의 해지)에 따라 위법계약이 '
 '해지되는 경우 회사가 적립한 해지 당시의 계약자적립액 및 미경과보험료를 반환하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 58},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000294',
              'chunk_char_len': 123,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
