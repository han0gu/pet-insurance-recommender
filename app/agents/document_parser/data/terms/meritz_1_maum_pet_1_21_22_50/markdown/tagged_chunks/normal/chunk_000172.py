from langchain_core.documents import Document

chunk = Document(
    page_content=('하는 별첨 신청서를 작성합니다.제3조(보험료의 영수)자동납입일자는 이 보험계약 청약서에 기재된 보험료 납입 해당일에도 불구하고 매월 '
 '회사\n'
 '가 정하는 날 중 보험계약자가 희망하는 일자로 합니다.제4조(계약 후 알릴 의무)계약자는 거래은행 지정계좌의 번호가 변경 또는 거래 '
 '정지된 경우에는 즉시 이 사실을 회'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000172',
              'chunk_char_len': 170,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
