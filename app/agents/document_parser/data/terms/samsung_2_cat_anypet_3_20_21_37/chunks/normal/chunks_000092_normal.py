from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 해지요구를 받은 날부터 10일 이내에 수락여부를 계약자에 통지하여야 하며, 거절할 때에 는 거절 사유를 함께 통지하여야 '
 '합니다. ③ 계약자는 회사가 정당한 사유 없이 제1항의 요구를 따르지 않는 경우 해당 계약을 해지할 수 있습 니다. ④ 제1항 및 '
 '제3항에 따라 계약이 해지된 경우 회사는 제30조(보험료의 환급) 제1항 제1호에 따른 환 급금을 계약자에게 지급합니다. ⑤ 계약자는 '
 '제1항에 따른 제척기간에도 불구하고 민법 등 관계 법령에서 정하는 바에 따라 법률상의 권리를 행사할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 17},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000092',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
