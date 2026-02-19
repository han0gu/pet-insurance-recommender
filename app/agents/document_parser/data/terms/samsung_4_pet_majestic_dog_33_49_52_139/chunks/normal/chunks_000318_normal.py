from langchain_core.documents import Document

chunk = Document(
    page_content=('• 보험료 납입기간 중 계약이 해지될 경우 해약환급금이 없고, 보험료 납입기간이 완료된 이후 계약이 해지될 경우 표준형 상품 대비 적은 '
 '해약환급금을 지급하는 대신 표준형 상품보다 낮은 보험료로 가입할 수 있도록 한 상품입니다. • 해약환급금을 계산할 때 기준이 되는 표준형 '
 '상품의 해약환급금은 “보험료 및 해약환급금 산출방법서”에 따라 계산한 금액으로 해지율을 적용하지 않고 계산합니다. • 회사는 계약을 '
 '체결할 때 표준형 상품의 보험료 및 해약환급금(환급률 포함) 수준을 비교∙안내해 드립니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 64},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000318',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
