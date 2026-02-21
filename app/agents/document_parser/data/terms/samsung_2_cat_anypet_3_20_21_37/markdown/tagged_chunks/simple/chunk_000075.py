from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항의 규정에 따라 해지하지 않은 계약은 파산선고 후 3개월이 지난 때에는 그 효력을 잃습니다.\n'
 '- ③ 제1항의 규정에 따라 계약이 해지되거나 제2항의 규정에 따라 계약이 효력을 잃는 경우에 회사는\n'
 '- 제30조(보험료의 환급)에 의한 보험료를 계약자에게 지급합니다.\n'
 '# 제29조(위법계약의 해지)① 계약자는 「금융소비자보호에 관한 법률」 제47조 및 관련규정이 정하는 바에 따라 계약체결에 대한\n'
 '회사의 법위반사항이 있는 경우 계약체결일부터 5년 이내의 범위에서 계약자가 위반사항을 안 날'),
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
 'indexing': {'chunk_id': 'chunk_000075',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
