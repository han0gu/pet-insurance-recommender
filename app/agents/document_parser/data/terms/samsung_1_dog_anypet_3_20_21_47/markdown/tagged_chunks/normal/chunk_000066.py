from langchain_core.documents import Document

chunk = Document(
    page_content=('담보권실행, 국세 및 지방세 체납처분절차에 의해 계약이 해지된 경우에는, 회사는 해지 당시의 피\n'
 '보험자가 계약자의 동의를 얻어 계약 해지로 회사가 채권자에게 지급한 금액을 회사에게 지급하고\n'
 '제19조(계약내용의 변경 등) 제1항의 절차에 따라 계약자 명의를 피보험자로 변경하여 계약의 특\n'
 '별부활(효력회복)을 청약할 수 있음을 피보험자에게 통지하여야 합니다.【강제집행】 사법상 또는 행정법상의 의무를 이행하지 않는 사람에 '
 '대하여 국가가 강제 권력으로 그 의무\n'
 '를 이행하는 것을 말합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000066',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
