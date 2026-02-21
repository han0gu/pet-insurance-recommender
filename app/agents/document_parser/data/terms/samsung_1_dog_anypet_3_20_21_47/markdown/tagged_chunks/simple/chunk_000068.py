from langchain_core.documents import Document

chunk = Document(
    page_content=('방세 체납시 국세청 및 지방자치단체에 의해 채무자의 환급금이 압류될 수 있으며, 체납처분 절차에 따라\n'
 '회사는 채권자에게 환급금을 지급하게 됩니다.- ② 회사는 제1항에 의한 계약자 명의변경 신청 및 계약의 특별부활(효력회복) 청약을 '
 '승낙하며, 계약\n'
 '- 은 청약한 때부터 특별부활(효력회복) 됩니다.\n'
 '- ③ 회사는 제1항의 통지를 계약이 해지된 날부터 7일 이내에 하여야 합니다. 다만, 회사의 통지가 7일\n'
 '- 을 지나서 도달하고 이후 피보험자가 제1항에 의한 계약자 명의변경 신청 및 계약의 특별부활(효'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000068',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
