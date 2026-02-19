from langchain_core.documents import Document

chunk = Document(
    page_content=('금을 지급하지 않으며, 보험료 납입면제사유 및 유사암 납입지원 사유가 발생한 경우 에 회사는 보험료 납입을 면제 또는 지원하지 않습니다. '
 '다만, 질병으로 인한 사망 또 는 진단확정된 질병으로 장해분류표에서 정한 장해지급률이 80% 이상에 해당하는 장 해상태가 되어 보험금 '
 '지급사유, 보험료 납입면제사유 또는 유사암 납입지원 사유가 발생한 경우에는 이를 적용하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 136},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000850',
              'chunk_char_len': 207,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
