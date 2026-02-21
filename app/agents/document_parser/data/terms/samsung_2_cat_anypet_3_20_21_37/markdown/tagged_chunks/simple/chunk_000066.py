from langchain_core.documents import Document

chunk = Document(
    page_content=('- 은 청약한 때부터 특별부활(효력회복) 됩니다.\n'
 '- ③ 회사는 제1항의 통지를 계약이 해지된 날부터 7일 이내에 하여야 합니다. 다만, 회사의 통지가 7일\n'
 '- 을 지나서 도달하고 이후 피보험자가 제1항에 의한 계약자 명의변경 신청 및 계약의 특별부활(효\n'
 '- 력회복)을 청약한 경우에는 계약이 해지된 날부터 7일이 되는 날에 특별부활(효력회복) 됩니다.\n'
 '- ④ 피보험자는 통지를 받은 날부터 15일 이내에 제1항의 절차를 이행할 수 있습니다.'),
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
 'indexing': {'chunk_id': 'chunk_000066',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
