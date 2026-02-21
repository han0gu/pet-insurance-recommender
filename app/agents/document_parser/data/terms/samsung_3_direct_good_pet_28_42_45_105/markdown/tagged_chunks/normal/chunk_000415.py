from langchain_core.documents import Document

chunk = Document(
    page_content=('수증 및 내시경영상검사결과지 등\n'
 '다. 이물제거(구토유도약물) 시행한 경우: 구토유도약물 처방이 명시된 동물병원\n'
 '진료비 내역서(진료항목이 기재되어 있는 명세서, 수의사 처방전 포함), 의료\n'
 '비 영수증 및 수의사처방전 등5. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발생 신분증, 본인이\n'
 '아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이\n'
 '확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)\n'
 '6. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류-'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000415',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
