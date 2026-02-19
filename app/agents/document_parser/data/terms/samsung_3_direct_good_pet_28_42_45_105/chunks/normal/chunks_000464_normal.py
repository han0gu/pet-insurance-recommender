from langchain_core.documents import Document

chunk = Document(
    page_content=('4.9만원\n'
 '- 보험금 미지급\n'
 '⑥ 제1항의 「연간」이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지의 기간을 의미합니다.\n'
 '제 2조 (보험금 지급에 관한 세부규정)\n'
 '보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못\n'
 '할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있습니다.\n'
 '제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 77},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000464',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
