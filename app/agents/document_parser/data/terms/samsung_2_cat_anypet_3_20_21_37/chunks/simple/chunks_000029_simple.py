from langchain_core.documents import Document

chunk = Document(
    page_content=('기 간 | 지 급 이 자\n'
 '지급기일의 다음 날부터 30일 이내 기간 | 보험계약대출이율\n'
 '지급기일의 31일이후부터 60일 이내 기간 | 보험계약대출0|율+가산이율(4.0%)\n'
 '지급기일의 61일이후부터 90일 이내 기간 | 보험계약대출이율+가산이율(6.0%)\n'
 '지급기일의 91일 이후 기간 | 보험계약대출이율+가산이율(8.0%)\n'
 '주) 보험계약대출이율은 보험개발원이 공시하는 보험계약대출이율을 적용합니다.\n'
 '제9조(보험금 등의 지급한도)'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 8},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000029',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
