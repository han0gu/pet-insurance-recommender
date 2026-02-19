from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하 「보험기간」 이라 합니다) 중에 상해 또는 진단확정된 질병으로 '
 '「깁스(Cast)치료」 (이하 「깁스치료」 라 합니다)를 받은 경우에는 매사고마다 보험증권에 기재된 이 특별약관의 보험가입금액 을 '
 '깁스치료비(부목치료 제외)로 보험수익자에게 지급합니다. ② 제1항의 깁스치료비(부목치료 제외)는 매사고마다 지급합니다. 다만, 동일한 '
 '상해사고 또는 질병으로 깁스치료를 2회이상 받거나 동시에 서로 다른 신체부위에 깁스치료를 받은 경우에는 1회에 한하여 보상합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 93},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000497',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
