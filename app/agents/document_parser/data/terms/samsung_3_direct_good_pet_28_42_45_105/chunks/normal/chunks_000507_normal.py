from langchain_core.documents import Document

chunk = Document(
    page_content=('제 2조 (보험금 지급에 관한 세부규정)\n'
 '보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못 할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 '
 '제3자의 의견에 따를 수 있습니다. 제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회 사가 전액 '
 '부담합니다.\n'
 '제 3조 (보험금을 지급하지 않는 사유)\n'
 '① 회사는 아래의 사유로 보험금 지급사유가 발생한 때에는 보험금을 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 82},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000507',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
