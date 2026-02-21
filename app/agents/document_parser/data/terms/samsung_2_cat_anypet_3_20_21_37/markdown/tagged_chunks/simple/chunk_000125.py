from langchain_core.documents import Document

chunk = Document(
    page_content=('있어야 합니다.# 제3조(보험가입금액)피보험자의 보험가입금액은 동일하게 책정하는 것을 원칙으로 합니다.제4조(피보험자의 증가, 감소 또는 '
 '교체)- ① 단체계약을 맺은 후 피보험자를 증가, 감소 또는 교체코자 하는 경우에는 계약자 또는 피보험자는\n'
 '- 지체없이 서면으로 그 사실을 회사에 알리고 회사의 승인을 받아야 합니다.\n'
 '- ② 이 계약기간 중 피보험자 감소의 경우는 당해 피보험자의 계약은 해지된 것으로 하며, 새로이 증가\n'
 '- 또는 교체되는 피보험자의 보험기간은 이 계약의 남은 보험기간으로 하고, 이로 인하여 발생되는'),
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
 'indexing': {'chunk_id': 'chunk_000125',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
