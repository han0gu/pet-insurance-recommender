from langchain_core.documents import Document

chunk = Document(
    page_content=('보험기간 중 계약이 해지될 경우 “보험료 및 해약환급금 산출방법서”에 따라 계산\n'
 '한 금액을 해약환급금으로 지급합니다.- 64 -5. 제1호, 제2호 및 제3호에서 표준형 상품이란 보험료 산출시 적용한 모든 기초율(다\n'
 '만, 해지율은 적용하지 않습니다)이 동일한 상품을 말하며, 해약환급금을 계산할\n'
 '때 기준이 되거나 비교∙안내를 위한 상품으로서 판매는 하지 않습니다.- ② 해약환급금의 지급사유가 발생한 경우 계약자는 회사에 '
 '해약환급금을 청구하여야 하\n'
 '- 며, 회사는 청구를 접수한 날부터 3영업일 이내에 해약환급금을 지급합니다. 해약환급'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000274',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
