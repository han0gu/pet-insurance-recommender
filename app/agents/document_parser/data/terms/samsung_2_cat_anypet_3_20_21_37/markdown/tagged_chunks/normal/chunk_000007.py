from langchain_core.documents import Document

chunk = Document(
    page_content=('더한 금액을 다음 1년의 원금으로 하는 이자 계산방법을 말합니다.【복리】 이자는 계산법에 따라 단리와 복리로 나눕니다. 단리는 원금에 '
 '대해서만 이자를 계산하는\n'
 '방법이고, 복리는 원금에 대한 이자를 원금에 가산한 후 이 합계액을 새로운 원금으로 계산하는 방\n'
 '법입니다.\n'
 '(예시) 원금 : 100원, 이자율 : 연 10%\n'
 '1년 후\n'
 '단리계산법 : 원금 + (원금×10%) = 110원\n'
 '복리계산법 : 원금 + (원금×10%) = 110원\n'
 '2년 후\n'
 '단리계산법 : 원금 + (원금×10%) + (원금×10%) = 120원'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000007',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
