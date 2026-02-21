from langchain_core.documents import Document

chunk = Document(
    page_content=('다.제5관 보험료의 납입# 제21 조(제1회 보험료 등 및 회사의 보장개시)- ① 회사는 계약의 청약을 승낙하고 제1회 보험료 등을 받은 '
 '때부터 이 약관이 정한 바에 따라 보장을\n'
 '- 합니다.\n'
 '- ② 회사가 계약자로부터 계약의 청약과 함께 제1회 보험료 등을 받은 경우에 그 청약을 승낙하기 전\n'
 '- 에 계약에서 정한 보험사고가 생긴 때에는 회사는 계약상의 보장을 합니다.\n'
 '- ③ 제2항의 규정에도 불구하고 회사는 다음 중 한 가지에 해당되는 경우에는 보장을 하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000052',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
