from langchain_core.documents import Document

chunk = Document(
    page_content=('⑦ 제1항에 따라 특별약관이 해지된 경우에는 이 특별약관의 해약환급금을 계약자에게 지급합니다.\n'
 '제22조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 74},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000425',
              'chunk_char_len': 92,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
