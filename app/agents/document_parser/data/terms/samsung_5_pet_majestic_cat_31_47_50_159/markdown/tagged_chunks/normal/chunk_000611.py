from langchain_core.documents import Document

chunk = Document(
    page_content=('- (치료비 세부내역 포함), 내시경영상검사결과지\n'
 '- 다. 이물제거(구토유도약물) 시행한 경우: 구토유도약물 처방이 명시된 동물병원\n'
 '- 진료비 영수증(치료비 세부내역 포함) 및 수의사처방전\n'
 '5. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발생 신분증, 본인이\n'
 '아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이\n'
 '확보된 전자적 수단을 활용한 피보험자 의사표시의 확인방법 포함)\n'
 '6. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류-'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000611',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
