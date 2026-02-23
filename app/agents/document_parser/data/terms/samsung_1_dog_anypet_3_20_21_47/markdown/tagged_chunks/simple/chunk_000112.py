from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 피해자로부터 손해배상책임에 관한 소송을 제기 받았을 경우\n'
 '② 계약자 또는 피보험자가 제1항 각호의 통지를 게을리 하여 손해가 증가된 때에는 회사는 그 증가\n'
 '된 손해를 보상하여 드리지 않으며, 제1항 제3호의 통지를 게을리 한 때에는 소송비용과 변호사비\n'
 '용도 보상하여 드리지 않습니다. 다만, 계약자 또는 피보험자가 상법 제657조 제1항에 의해 보험\n'
 "사고의 발생을 회사에 알린 경우에는 제1조(보상하는 손해) 제1호 및 제2호 '다'목 또는 '라'목의"),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000112',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
