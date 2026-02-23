from langchain_core.documents import Document

chunk = Document(
    page_content=('- 합니다)를 받은 경우에는 매사고마다 보험증권에 기재된 이 특별약관의 보험가입금액\n'
 '- 을 깁스치료비(부목치료 제외)로 보험수익자에게 지급합니다.\n'
 '- ② 제1항의 깁스치료비(부목치료 제외)는 매사고마다 지급합니다. 다만, 동일한 상해사고\n'
 '- 또는 질병으로 깁스치료를 2회이상 받거나 동시에 서로 다른 신체부위에 깁스치료를\n'
 '- 받은 경우에는 1회에 한하여 보상합니다.\n'
 '# 제2조 (보험금 지급에 관한 세부규정)보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000419',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
