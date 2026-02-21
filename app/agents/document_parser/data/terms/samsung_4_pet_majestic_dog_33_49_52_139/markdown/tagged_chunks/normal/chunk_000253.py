from langchain_core.documents import Document

chunk = Document(
    page_content=('보험금 지급사유를 이유로 부활(효력회복)을 거절하지 않습니다.\n'
 '③ 제1항에서 정한 계약의 부활(효력회복)이 이루어진 경우라도 계약자 또는 피보험자가\n'
 '최초계약 청약시(2회 이상 부활이 이루어진 경우 종전 모든 부활 청약 포함) 제15조\n'
 '(계약 전 알릴 의무)를 위반한 경우에는 제17조(알릴 의무 위반의 효과)가 적용됩니\n'
 '다.- \n'
 '# 제31조 (강제집행 등으로 인하여 해지된 특별약관의 특별부활(효력회복))① 회사는 계약자의 해약환급금 청구권에 대한 강제집행, '
 '담보권실행, 국세 및 지방세 체'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000253',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
