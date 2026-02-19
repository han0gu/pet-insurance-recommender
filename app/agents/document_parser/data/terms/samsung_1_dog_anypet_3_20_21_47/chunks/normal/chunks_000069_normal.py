from langchain_core.documents import Document

chunk = Document(
    page_content=('【납입기일】 계약자가 제2회 이후의 보험료를 납입하기로 한 날을 말합니다.\n'
 '제23조[보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지]'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 14},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000069',
              'chunk_char_len': 81,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
