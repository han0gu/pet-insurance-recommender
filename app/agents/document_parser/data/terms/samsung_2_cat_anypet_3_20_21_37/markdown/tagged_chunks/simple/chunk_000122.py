from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이내에 회사에 납입하여야 합니다.\n'
 '- ④ 제1항 내지 제2항의 통지는 서면(전자적 수단을 포함합니다)으로 이루어져야 합니다.\n'
 '# 제5조(자료의 제출 및 열람)① 계약자는 계약이 효력상실 또는 해지된 경우에는 효력상실 또는 해지일까지의 보험료를 확정하기\n'
 '위하여 필요한 서류를 효력상실 또는 해지 즉시 회사에 제출해야 합니다.30 -당신에게 좋은보험 삼성화재회사는 보험기간 중이나 보험기간 '
 '만료 후 1년 이내에는 보험료 계산에 필요한 경우에 계약자의'),
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
 'indexing': {'chunk_id': 'chunk_000122',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
