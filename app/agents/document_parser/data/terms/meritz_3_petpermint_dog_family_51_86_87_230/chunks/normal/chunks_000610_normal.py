from langchain_core.documents import Document

chunk = Document(
    page_content=('제12조(계약 후 알릴 의무)\n'
 '\uf000 계약을 맺은 후 아래와 같은 사실이 생긴 경우에는 계약 자 또는 피보험자는 지체없이 서면으로 회사에 알리고 보험 증권에 '
 '확인을 받아야 합니다.\n'
 '① 청약서의 기재사항을 변경하고자 할 때 또는 변경이 생겼음을 알았을 때 ② 이 계약에서 보장하는 위험과 동일한 위험을 보장하는 계약을 '
 '다른 보험자와 맺으려고 하든지 또는 이와 같 은 계약이 있음을 알았을 때 ③ 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 181},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000610',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
