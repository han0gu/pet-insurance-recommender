from langchain_core.documents import Document

chunk = Document(
    page_content=('- 손해를 배상할 책임을 집니다.\n'
 '- ③ 회사가 보험금 지급여부 및 지급금액에 관하여 현저하게 공정을 잃은 합의로 보험수\n'
 '- 익자에게 손해를 가한 경우에도 회사는 제2항에 따라 손해를 배상할 책임을 집니다.\n'
 '# <용어풀이># [현저하게 공정을 잃은 합의]회사가 보험수익자의 경제적․신체적․정신적인 어려움, 경솔함, 경험 부족 등을 이용하여 '
 '동일․유사'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000145',
              'chunk_char_len': 195,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
