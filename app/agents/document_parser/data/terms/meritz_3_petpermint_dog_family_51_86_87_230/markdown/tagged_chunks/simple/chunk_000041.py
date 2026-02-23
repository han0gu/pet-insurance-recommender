from langchain_core.documents import Document

chunk = Document(
    page_content=('에게도 효력이 미칩니다.\n'
 '\uf000 계약자가 2명 이상인 경우에는 그 책임을 연대로 합니\n'
 '다.# 【계약자가 2명 이상인 경우 】계약자가 2명 이상인 경우, 계약 전 알릴 의무, 보험료\n'
 '납입의무 등 보험계약에 따른 계약자의 의무를 연대로\n'
 '합니다.# 【연대】2인 이상이 공동으로 책임지는 것을 뜻하며, 각자가 채\n'
 '무의 전부를 이행할 책임을 지되(지분만큼 분할하여 책\n'
 '임을 지는 것과는 다름), 다만 어느 1인의 이행으로 나\n'
 '머지 사람들도 책임을 면하게 되는 것을 말합니다.# 제3관 계약자의 계약 전 알릴 의무 등제15조(계약 전 알릴 의무)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000041',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
