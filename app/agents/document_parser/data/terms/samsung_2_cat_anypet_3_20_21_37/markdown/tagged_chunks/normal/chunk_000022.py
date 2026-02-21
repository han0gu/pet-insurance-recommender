from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확보된 전자적 수단을 활용\n'
 '- 한 피보험자 의사표시의 확인방법 포함)\n'
 '② 회사는 제1항에 열거하는 서류 이외의 서류 제출을 요구할 수 있습니다.당신에게 좋은보험 삼성화재- 7 -# 제8조(보험금의 '
 '지급절차)① 회사는 제7조(보험금의 청구)에서 정한 서류를 접수한 때에는 접수증을 교부하고, 그 서류를 접수\n'
 '받은 후 지체없이 지급할 보험금을 결정하고 지급할 보험금이 결정되면 7일 이내에 이를 지급하여'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000022',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
