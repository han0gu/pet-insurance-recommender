from langchain_core.documents import Document

chunk = Document(
    page_content=('여야 합니다.- 1. 보험금 청구서(회사 양식)\n'
 '- 2. 사망을 확인할 수 있는 서류(동물폐사확인서, 동물화장증명서 등)\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발생 신분증, 본인이 아닌 경우에\n'
 '- 는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확보된 전자적 수단을 활용\n'
 '- 한 피보험자 의사표시의 확인방법 포함)\n'
 '- ② 회사는 제1항에 열거하는 서류 이외의 서류 제출을 요구할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000131',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
