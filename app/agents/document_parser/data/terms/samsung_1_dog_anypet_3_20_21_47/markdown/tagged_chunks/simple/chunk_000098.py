from langchain_core.documents import Document

chunk = Document(
    page_content=('정한 1일당 수술비확장 보상한도액을 한도로 아래의 산식에 따라 보상합니다.【수술비용 확대보장 보험금】 아래 ①과 ② 중 적은 금액- '
 '(피보험자가 부담한 수술당일치료비 - 보통약관에서 지급한 치료비보험금) × 보상비율\n'
 '- ② 보험증권에서 정한 1일당 수술비용 확대보장 보상한도액\n'
 '② 보험기간 중에 발생한 사고로 회사가 지급하는 수술비용 확대보장 보험금의 총 한도는 보험증권에'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000098',
              'chunk_char_len': 208,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
