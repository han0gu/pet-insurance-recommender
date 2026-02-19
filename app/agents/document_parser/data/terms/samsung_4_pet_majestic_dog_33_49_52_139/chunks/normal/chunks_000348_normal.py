from langchain_core.documents import Document

chunk = Document(
    page_content=('관계없이 신체의 장해정도에 따라 장해분류표의 구분에 준하여 지급액을 결정합니다. 다만, 장해분류표의 각 장해분류별 최저 지급률 장해정도에 '
 '이르지 않는 후유장해에 대하여는 상해 후유장해보험금을 지급하지 않습니다.\n'
 '④ 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못할 때는 보험수익자와 회사가 함께 제3자를 정하고 '
 '그 제3자의 의견에 따를 수 있 습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하 며, 보험금 지급사유 '
 '판정에 드는 의료비용은 회사가 전액 부담합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 69},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000348',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
