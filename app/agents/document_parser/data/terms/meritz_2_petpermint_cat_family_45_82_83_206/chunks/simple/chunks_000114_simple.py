from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제3조(보험금의 지급사유)에서 정한 일반상해80%이상후 유장해보험금 지급사유가 발생한 경우에는 이 보장책임은 그 때부터 '
 '소멸됩니다. \uf000 제1항에 따라 이 계약의 보장책임이 소멸된 때에는 회사 는 이 보장책임의 해약환급금을 지급하지 않으며, 그 때까 '
 '지「보험료 및 해약환급금 산출방법서」에서 정하는 바에 따라 회사가 적립한 적립부분의 계약자적립액(중도인출이 있는 경우에는 중도인출 원금과 '
 '이자를 차감하고 적립한 금 액을 말합니다) 및 미경과보험료를 계약자에게 지급합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 70},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000114',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
