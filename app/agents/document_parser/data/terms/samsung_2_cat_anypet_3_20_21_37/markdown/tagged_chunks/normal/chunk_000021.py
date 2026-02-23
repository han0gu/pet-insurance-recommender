from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 보상하여 드리지 않습니다.\n'
 '- ③ 회사가 위 제1항에 대한 손해의 사실을 확인하기 어려운 경우에는 계약자 또는 피보험자에게 필요\n'
 '- 한 증거자료의 제출을 요구할 수 있습니다.\n'
 '# 제7조(보험금의 청구)① 피보험자가 보험금을 청구할 때에는 다음의 서류를 회사에 제출하여야 합니다.- 1. 보험금 청구서(회사 '
 '양식)\n'
 '- 2. 진료비 내역서(진료항목이 기재되어 있는 명세서, 수의사 처방전 포함) 및 치료비 영수증\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발생 신분증, 본인이 아닌 경우에'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000021',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
