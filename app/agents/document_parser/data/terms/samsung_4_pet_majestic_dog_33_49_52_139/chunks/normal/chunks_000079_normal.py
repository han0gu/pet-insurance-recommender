from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제1항에 따라 계약을 해지하였을 때에는 제36조(해약환급금) 제1항에 따른 해약환급 금을 계약자에게 지급합니다. ④ 제1항 제1호에 '
 '의한 계약의 해지가 보험금 지급사유 발생 후에 이루어진 경우에 회사 는 보험금을 지급하지 않으며, 계약 전 알릴 의무 위반 사실(계약해지 '
 '등의 원인이 되 는 위반사실을 구체적으로 명시) 뿐만 아니라 계약 전 알릴 의무사항이 중요한 사항에 해당되는 사유를 "반대증거가 있는 '
 '경우 이의를 제기할 수 있습니다"라는 문구와 함께 계약자에게 서면 또는 전자문서 등으로 알려드립니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 40},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000079',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
