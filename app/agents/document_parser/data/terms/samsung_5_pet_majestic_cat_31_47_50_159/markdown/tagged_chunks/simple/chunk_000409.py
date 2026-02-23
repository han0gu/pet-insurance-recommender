from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 감염병 발생 신고(보고)서\n'
 '3. 사고증명서(진단서, 진료비계산서, 사망진단서, 장해진단서, 입원치료확인서, 의사\n'
 '처방전(처방조제비) 등)\n'
 '4. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증, 본인이\n'
 '아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이- \n'
 '- 88 -확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)5. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 '
 '서류② 제1항 제3호의 사고증명서는 의료법 제3조(의료기관)에 규정한 국내의 병원이나 의원'),
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
 'indexing': {'chunk_id': 'chunk_000409',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
