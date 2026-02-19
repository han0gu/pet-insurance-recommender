from langchain_core.documents import Document

chunk = Document(
    page_content=('제9조(보험금 등의 지급한도)\n'
 '① 제4조(보상하는 손해)에서 정한 치료비보험금은 제4조의 사고로 발생한 치료비에서 아래의 제2항 과 제3항에 따른 적용 자기부담금을 '
 '차감한 후 보험증권에 기재된 보상비율(%)을 곱한 금액이며, 아래의 제2항과 제3항의 적용 지급한도액을 한도로 보상하여 드립니다.\n'
 '【치료비보험금】 아래 ①과 ② 중 적은 금액\n'
 '① (피보험자가 부담한 치료비 - 적용 자기부담금) × 보상비율 ② 적용 지급한도액\n'
 '② 입원 또는 입원 중 수술이 이루어진 경우의 적용 지급한도액 및 적용 자기부담금은 아래와 같습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 8},
 'term_type': 'basic',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000030',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
