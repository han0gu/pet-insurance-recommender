from langchain_core.documents import Document

chunk = Document(
    page_content=('제32조 (강제집행 등으로 인하여 해지된 계약의 특별부활(효력회복))\n'
 '① 회사는 계약자의 해약환급금 청구권에 대한 강제집행, 담보권실행, 국세 및 지방세 체 납처분절차에 따라 계약이 해지된 경우 해지 당시의 '
 '보험수익자가 계약자의 동의를 얻어 계약 해지로 회사가 채권자에게 지급한 금액을 회사에 지급하고 제24조(계약내 용의 변경 등) 제1항의 '
 '절차에 따라 계약자 명의를 보험수익자로 변경하여 계약의 특 별부활(효력회복)을 청약할 수 있음을 보험수익자에게 통지하여야 합니다.\n'
 '<용어풀이>\n'
 '[강제집행과 담보권실행]'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 45},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000141',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
