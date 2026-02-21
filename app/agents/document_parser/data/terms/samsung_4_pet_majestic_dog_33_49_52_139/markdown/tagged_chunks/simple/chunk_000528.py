from langchain_core.documents import Document

chunk = Document(
    page_content=('활(효력회복)일을 보험계약일로 하여 제3조(보험금의 지급사유) 제3항을 적용합니다.# 제23조 (강제집행 등으로 인하여 해지된 특별약관의 '
 '특별부활(효력회복))① 회사는 계약자의 해약환급금 청구권에 대한 강제집행, 담보권실행, 국세 및 지방세 체\n'
 '납처분절차에 의해 특별약관이 해지된 경우 해지 당시의 보험수익자가 계약자의 동의\n'
 '를 얻어 특별약관 해지로 회사가 채권자에게 지급한 금액을 회사에 지급하고 제17조\n'
 '(특별약관내용의 변경 등) 제1항의 절차에 따라 계약자 명의를 보험수익자로 변경하'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000528',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
