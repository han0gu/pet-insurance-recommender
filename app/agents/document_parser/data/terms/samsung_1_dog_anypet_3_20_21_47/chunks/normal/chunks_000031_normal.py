from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 제1항에 열거하는 서류 이외의 서류 제출을 요구할 수 있습니다.\n'
 '제8조(보험금의 지급절차)\n'
 '① 회사는 제7조(보험금의 청구)에서 정한 서류를 접수한 때에는 접수증을 교부하고, 그 서류를 접수 받은 후 지체없이 지급할 보험금을 '
 '결정하고 지급할 보험금이 결정되면 7일 이내에 이를 지급하여 드립니다. 또한, 지급할 보험금이 결정되기 전이라도 피보험자의 청구가 있을 '
 '때에는 회사가 추정 한 보험금의 50% 상당액을 가지급보험금으로 지급합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 8},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000031',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
