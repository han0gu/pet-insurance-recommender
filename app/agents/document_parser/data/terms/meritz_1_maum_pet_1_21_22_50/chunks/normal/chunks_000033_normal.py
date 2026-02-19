from langchain_core.documents import Document

chunk = Document(
    page_content=('① 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.\n'
 '1. 청구서(회사양식) 2. 사고증명서(동물병원 진료비 영수증, 동물병원 진료비세부내역서(진료 항목별 영수금 액 포함), 동물병원 '
 '진료기록부, X-ray 등 방사선 촬영을 하는 경우 해당 사진(촬영 일자 및 시간 필수) 등) 3. 신분증(주민등록증이나 운전면허증 등 '
 '사진이 붙은 정부기관 발행 신분증, 본인이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확 보된 전자적 '
 '수단을 활용한 피보험자 의사표시의 확인방법 포함)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000033',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
