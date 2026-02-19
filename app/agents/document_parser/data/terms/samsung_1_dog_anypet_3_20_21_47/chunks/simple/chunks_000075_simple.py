from langchain_core.documents import Document

chunk = Document(
    page_content=('5. 제3호 및 제4호의 내용에 관한 사항을 계약자에게 안내할 것\n'
 '⑤ 제1항에 따라 계약이 해지된 경우에는 제30조(보험료의 환급)에 따라 보험료를 계약자에게 지급합 니다.\n'
 '제24조[보험료의 납입연체로 인한 해지계약의 부활(효력회복)]'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 15},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000075',
              'chunk_char_len': 131,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
