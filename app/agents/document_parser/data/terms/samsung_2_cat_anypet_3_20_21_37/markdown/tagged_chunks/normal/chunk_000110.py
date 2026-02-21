from langchain_core.documents import Document

chunk = Document(
    page_content=('- 질성이 확보되어야 합니다.\n'
 '- ② 단체 구성원의 일부만을 대상으로 가입하는 경우에는 대상단체의 위험과 피보험단체의 위험의 동\n'
 '- 질성이 유지되어야 합니다.\n'
 '- 27 -당신에게 좋은보험 삼성화재제4조(보험의 목적의 증가 감소 또는 교체)- ① 계약을 맺은 후 보험의 목적을 증가, 감소 또는 '
 '교체코자 하는 경우에는 계약자 또는 피보험자는\n'
 '- 지체없이 서면으로 그 사실을 회사에 알리고 회사의 승인을 받아야 합니다.\n'
 '- ② 이 계약기간 중 보험의 목적 감소의 경우는 당해 보험의 목적의 계약은 해지된 것으로 하며 새로이'),
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
 'indexing': {'chunk_id': 'chunk_000110',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
