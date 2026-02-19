from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약을 맺은 후 보험의 목적을 증가, 감소 또는 교체코자 하는 경우에는 계약자 또는 피보험자는 지체없이 서면으로 그 사실을 회사에 '
 '알리고 회사의 승인을 받아야 합니다. ② 이 계약기간 중 보험의 목적 감소의 경우는 당해 보험의 목적의 계약은 해지된 것으로 하며 새로이 '
 '증가 또는 교체되는 보험의 목적의 보험기간은 이 계약의 남은 보험기간으로 하고, 이로 인하여 발 생되는 추가 또는 환급보험료는 일단위로 '
 '계산하여 받거나 돌려 드립니다'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 36},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000179',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
