from langchain_core.documents import Document

chunk = Document(
    page_content=('제9조(보험금 등의 지급한도)\n'
 '① 제4조(보상하는 손해)에서 정한 치료비보험금은 보험증권에 기재된 자기부담금을 차감 후 보상비율 을 곱한 금액이며 보험증권에 정한 '
 '1일당 보상한도액을 한도로 보상합니다. 자기부담금은 1일당 치료비에서 차감합니다. ② 보험기간 중에 발생한 사고로 회사가 지급하는 '
 '치료비보험금의 총 합계는 보험증권에 기재된 총보 상한도액을 한도로 합니다.\n'
 '【치료비보험금】 아래 ①과 ② 중 적은 금액\n'
 '(피보험자가 부담한 1일당 치료비 - 1일당 자기부담금) × 보상비율 ② 보험증권에서 정한 1일당 보상한도액'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 9},
 'term_type': 'basic',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000035',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
