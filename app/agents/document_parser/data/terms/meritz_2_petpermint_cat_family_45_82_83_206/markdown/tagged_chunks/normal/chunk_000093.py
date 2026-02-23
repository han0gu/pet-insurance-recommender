from langchain_core.documents import Document

chunk = Document(
    page_content=('는 이 보장책임의 해약환급금을 지급하지 않으며, 그 때까\n'
 '지「보험료 및 해약환급금 산출방법서」에서 정하는 바에\n'
 '따라 회사가 적립한 적립부분의 계약자적립액(중도인출이\n'
 '있는 경우에는 중도인출 원금과 이자를 차감하고 적립한 금\n'
 '액을 말합니다) 및 미경과보험료를 계약자에게 지급합니다.\n'
 '\uf000 피보험자가 사망한 경우에는 이 계약은 소멸되며, 이 경\n'
 '우 회사는 그 때까지「보험료 및 해약환급금 산출방법서」\n'
 '에서 정한 사망 당시 계약자적립액(중도인출이 있는 경우\n'
 '중도인출 원금과 이자를 차감하고 적립한 금액을 말합니다)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000093',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
