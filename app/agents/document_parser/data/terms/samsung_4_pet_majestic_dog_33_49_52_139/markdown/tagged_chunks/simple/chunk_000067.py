from langchain_core.documents import Document

chunk = Document(
    page_content=('- 5. 보험설계사 등이 계약자 또는 피보험자에게 알릴 기회를 주지 않았거나 계약자 또\n'
 '- 는 피보험자가 사실대로 알리는 것을 방해한 경우, 계약자 또는 피보험자에게 사실\n'
 '- 대로 알리지 않게 하였거나 부실한 사항을 알릴 것을 권유했을 때. 다만, 보험설계\n'
 '- 사 등의 행위가 없었다 하더라도 계약자 또는 피보험자가 사실대로 알리지 않거나\n'
 '- 부실한 사항을 알렸다고 인정되는 경우에는 계약을 해지할 수 있습니다.\n'
 '- ③ 제1항에 따라 계약을 해지하였을 때에는 제36조(해약환급금) 제1항에 따른 해약환급\n'
 '- 금을 계약자에게 지급합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000067',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
