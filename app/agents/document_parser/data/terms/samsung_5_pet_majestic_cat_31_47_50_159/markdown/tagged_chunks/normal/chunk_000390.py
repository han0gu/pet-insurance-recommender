from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 피보험자의 배우자 및 직계존비속에 의한 손해\n'
 '- 3. 피보험자가 범죄행위를 하던 중 또는 폭력행위 등 처벌에 관한 법률 제4조의 범죄\n'
 '- 단체를 구성 또는 이에 가담함으로써 발생된 손해\n'
 '- 4. 전쟁, 폭동, 소요, 노동쟁의 또는 이와 유사한 사변 중에 생긴 손해\n'
 '- 5. 피보험자와 고용관계에 있는 고용주 내지 고용상의 관리책임이 있는 자에 의해 발\n'
 '- 생한 손해\n'
 '- - 84 -\n'
 '# 제5조 (보험금의 청구)① 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.- 1. 보험금 청구서(회사양식)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000390',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
