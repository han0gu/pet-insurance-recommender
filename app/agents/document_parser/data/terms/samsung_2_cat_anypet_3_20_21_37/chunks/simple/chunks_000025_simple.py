from langchain_core.documents import Document

chunk = Document(
    page_content=('① 피보험자가 보험금을 청구할 때에는 다음의 서류를 회사에 제출하여야 합니다.\n'
 '1. 보험금 청구서(회사 양식) 2. 진료비 내역서(진료항목이 기재되어 있는 명세서, 수의사 처방전 포함) 및 치료비 영수증 3. '
 '신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발생 신분증, 본인이 아닌 경우에 는 본인의 인감증명서, 본인서명사실확인서 또는 '
 '안전성과 신뢰성이 확보된 전자적 수단을 활용 한 피보험자 의사표시의 확인방법 포함)\n'
 '② 회사는 제1항에 열거하는 서류 이외의 서류 제출을 요구할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 7},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000025',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
