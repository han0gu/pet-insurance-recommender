from langchain_core.documents import Document

chunk = Document(
    page_content=('【국세 및 지방세 체납처분 절차】 국세 또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하여 체\n'
 '납된 세금에 대하여 가산금 징수, 독촉장 발부 및 재산 압류 등의 집행을 하는 것을 말합니다. 국세 및 지\n'
 '방세 체납시 국세청 및 지방자치단체에 의해 채무자의 환급금이 압류될 수 있으며, 체납처분 절차에 따라\n'
 '회사는 채권자에게 환급금을 지급하게 됩니다.- ② 회사는 제1항에 의한 계약자 명의변경 신청 및 계약의 특별부활(효력회복) 청약을 '
 '승낙하며, 계약\n'
 '- 은 청약한 때부터 특별부활(효력회복) 됩니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000065',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
