from langchain_core.documents import Document

chunk = Document(
    page_content=('① 타인을 위한 계약의 경우 제33조(보험료의 환급)에 따른 계약자의 환급금 청구권에 대 한 강제집행, 담보권실행, 국세 및 지방세 '
 '체납처분절차에 따라 계약이 해지된 경우에 는, 회사는 해지 당시의 보험수익자가 계약자의 동의를 얻어 계약 해지로 회사가 채권 자에게 '
 '지급한 금액을 회사에게 지급하고 제23조(계약내용의 변경 등) 제1항의 절차에 따라 계약자 명의를 보험수익자로 변경하여 계약의 '
 '특별부활(효력회복)을 청약할 수 있 음을 보험수익자에게 통지하여야 합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 17},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000109',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
