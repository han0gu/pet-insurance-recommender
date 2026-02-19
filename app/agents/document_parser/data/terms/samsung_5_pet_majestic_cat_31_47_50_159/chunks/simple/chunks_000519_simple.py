from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조 (보험금의 청구)\n'
 '① 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.\n'
 '1. 청구서(회사양식) 2. 사고증명서(진단서, 진료비세부내역서("건강보험심사평가원 진료수가코드(EDI)" 필 수 기재), 진료기록부, '
 '수술증명서 등) 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증, 본인이 아닌 경우에는 본인의 인감증명서, '
 '본인서명사실확인서 또는 안전성과 신뢰성이 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함) 4. 기타 보험수익자가 '
 '보험금의 수령에 필요하여 제출하는 서류'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 94},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000519',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
