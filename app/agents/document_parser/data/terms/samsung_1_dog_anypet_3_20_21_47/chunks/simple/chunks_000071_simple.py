from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 납입최고(독촉)기간 내에 연체보험료를 납입하여야 한다는 내용 2. 납입최고(독촉)기간이 끝나는 날까지 보험료를 납입하지 않을 경우그 '
 '끝나는 날의 다음날에 계 약이 해지된다는 내용'),
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
 'indexing': {'chunk_id': 'chunk_000071',
              'chunk_char_len': 102,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
