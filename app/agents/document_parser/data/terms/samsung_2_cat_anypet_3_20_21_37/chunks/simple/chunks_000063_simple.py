from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 계약자가 제1회 보험료 등을 자동이체 또는 신용카드로 납입하는 경우에는 자동이체신청 및 신용 카드매출 승인에 필요한 정보를 회사에 '
 '제공한 때가 제1회 보험료 등을 납입한 때가 되나, 계약자 의 책임있는 사유로 자동이체 또는 매출승인이 불가능한 경우에는 제1회 보험료 '
 '등이 납입되지 않 은 것으로 봅니다. ⑤ 계약이 갱신되는 경우에는 제1항 내지 제3항에 의한 보장은 기존 계약에 의한 보장이 종료하는 때 '
 '부터 적용합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 13},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000063',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
