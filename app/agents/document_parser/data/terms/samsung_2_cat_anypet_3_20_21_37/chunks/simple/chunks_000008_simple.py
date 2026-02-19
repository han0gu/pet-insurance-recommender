from langchain_core.documents import Document

chunk = Document(
    page_content=('【복리】 이자는 계산법에 따라 단리와 복리로 나눕니다. 단리는 원금에 대해서만 이자를 계산하는 방법이고, 복리는 원금에 대한 이자를 '
 '원금에 가산한 후 이 합계액을 새로운 원금으로 계산하는 방 법입니다'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 5},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000008',
              'chunk_char_len': 111,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
