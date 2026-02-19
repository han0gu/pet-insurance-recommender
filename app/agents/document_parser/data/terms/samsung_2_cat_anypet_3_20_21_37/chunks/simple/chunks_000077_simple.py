from langchain_core.documents import Document

chunk = Document(
    page_content=('【강제집행】 사법상 또는 행정법상의 의무를 이행하지 않는 사람에 대하여 국가가 강제 권력으로 그 의무 를 이행하는 것을 말합니다. '
 '【담보권 실행】 담보권을 설정한 채권자가 채무를 이행하지 않는 채무자에 대하여 해당 담보권을 실행하 는 것을 말합니다. 법원은 채권자의 '
 '신청에 따른 강제집행 및 담보권실행으로 채무자의 환급금을 압류할 수 있으며, 법원의 추심명령 또는 전부명령에 따라 회사는 채권자에게 '
 '환급금을 지급하게 됩니다'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 15},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000077',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
