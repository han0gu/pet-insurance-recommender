from langchain_core.documents import Document

chunk = Document(
    page_content=('• 보험료 납입기간 중 계약이 해지될 경우 표준형 상품 대비 적은 해약환급금을 지급하는 대신 표준형 상품보다 낮은 보험료로 가입할 수 '
 '있도록 한 상품입니다. • 해약환급금을 계산할 때 기준이 되는 표준형 상품의 해약환급금은 “보험료 및 해약환급금 산출방법서”에 따라 '
 '계산한 금액으로 해지율을 적용하지 않고 계산합니다. • 회사는 계약을 체결할 때 표준형 상품의 보험료 및 해약환급금(환급률 포함) 수준을 '
 '비교∙ 안내해 드립니다. • 보험료 납입기간이란 계약을 체결할 때 보험료를 납입하기로 한 기간을 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 62},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000311',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
