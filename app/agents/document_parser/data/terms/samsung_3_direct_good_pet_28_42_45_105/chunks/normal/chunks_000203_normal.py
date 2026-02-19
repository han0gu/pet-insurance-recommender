from langchain_core.documents import Document

chunk = Document(
    page_content=('부실한 사항을 알렸다고 인정되는 경우에는 특별약관을 해지할 수 있습니다.\n'
 '③ 제1항에 따라 특별약관을 해지하였을 때에는 제35조(해약환급금)제1항에 따른 해약환\n'
 '급금을 계약자에게 지급합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 50},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000203',
              'chunk_char_len': 106,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
