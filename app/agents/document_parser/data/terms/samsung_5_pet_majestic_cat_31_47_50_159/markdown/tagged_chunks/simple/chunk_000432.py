from langchain_core.documents import Document

chunk = Document(
    page_content=('인하여 창상봉합술 치료를 받은 경우에도 보장하는 수가코드를 확인할 수 있는 경우 회사는 창상\n'
 '봉합술 치료비를 지급합니다.② 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지\n'
 '못할 때에는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수\n'
 '있습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000432',
              'chunk_char_len': 207,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
