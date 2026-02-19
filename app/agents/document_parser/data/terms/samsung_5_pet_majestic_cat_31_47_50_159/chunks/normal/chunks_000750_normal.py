from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 반려묘 의료비 확대보장(이물제거 특정처치)(연간2회한)(재가입형) 2. 반려묘 의료비 '
 '확대보장(MRI,CT)(연간1회한)(재가입형)\n'
 '제2조 (보험금 지급에 관한 세부규정)\n'
 '보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못 할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 '
 '제3자의 의견에 따를 수 있습니다. 제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회 사가 전액 '
 '부담합니다.\n'
 '제3조 (자기공명영상(MRI) 및 컴퓨터단층촬영(CT)의 정의)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 118},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000750',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
